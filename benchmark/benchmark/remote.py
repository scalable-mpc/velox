# Copyright(C) Facebook, Inc. and its affiliates.
from collections import OrderedDict
from fabric import Connection, ThreadingGroup as Group
from fabric.exceptions import GroupException
from paramiko import RSAKey
from paramiko.ssh_exception import PasswordRequiredException, SSHException
from os.path import basename, splitext
from time import sleep
from math import ceil
from copy import deepcopy
import subprocess
import time

from benchmark.config import Committee, Key, NodeParameters, BenchParameters, ConfigError
from benchmark.utils import BenchError, Print, PathMaker, progress_bar
from benchmark.progress import (
    RemoteSteps, StepError, failures, report_capture, report_failures
)
from benchmark.commands import CommandMaker
from benchmark.logs import LogParser, ParseError
from benchmark.instance import InstanceManager


class FabricError(Exception):
    ''' Wrapper for Fabric exception with a meaningfull error message. '''

    def __init__(self, error):
        assert isinstance(error, GroupException)
        message = list(error.result.values())[-1]
        super().__init__(message)


class ExecutionError(Exception):
    pass


class Bench:
    # Two options ride on every apt call. DPkg::Lock::Timeout bounds the wait
    # for a held lock at 10 minutes -- apt >= 2.0 otherwise waits forever, and
    # a fresh image runs apt-daily/unattended-upgrades at boot, which is what
    # used to stall the first apt call with nothing on screen.
    # DEBIAN_FRONTEND is set on the sudo invocation rather than exported by a
    # prelude, because sudoers runs with env_reset and would strip it. stdin is
    # closed on every step, so a debconf or needrestart prompt would hang until
    # the step timeout.
    APT = ('sudo DEBIAN_FRONTEND=noninteractive '
           'apt-get -o DPkg::Lock::Timeout=600 -y')

    # The step whose stdout is worth keeping: it reports what it killed.
    KILL_LOCKS = 'kill apt lock holders'

    # The four files apt and dpkg lock.
    LOCK_FILES = ('/var/lib/dpkg/lock-frontend /var/lib/dpkg/lock '
                  '/var/lib/apt/lists/lock /var/cache/apt/archives/lock')

    def __init__(self, ctx):
        self.manager = InstanceManager.make()
        self.settings = self.manager.settings
        try:
            ctx.connect_kwargs.pkey = RSAKey.from_private_key_file(
                self.manager.settings.key_path
            )
            self.connect = ctx.connect_kwargs
        except (IOError, PasswordRequiredException, SSHException) as e:
            raise BenchError('Failed to load SSH key', e)

    def _check_stderr(self, output):
        if isinstance(output, dict):
            for x in output.values():
                if x.stderr:
                    raise ExecutionError(x.stderr)
        else:
            if output.stderr:
                raise ExecutionError(output.stderr)

    @classmethod
    def _apt_unlock_steps(cls):
        ''' Steps that leave apt usable: wait out the boot-time apt activity,
        kill anything still holding the locks, then repair dpkg in case the
        thing we killed was halfway through a transaction. '''
        return [
            # cloud-init and the apt-daily units legitimately own apt for the
            # first minutes of a machine's life. Let them finish and stop them
            # from starting again, rather than killing them mid-transaction.
            ('wait for boot apt',
             'sudo cloud-init status --wait || true; '
             'sudo systemctl stop apt-daily.timer apt-daily-upgrade.timer '
             'apt-daily.service apt-daily-upgrade.service '
             'unattended-upgrades.service 2>/dev/null || true'),

            # Whatever still holds a lock after that is not going to let go.
            # fuser is exact but lives in psmisc, which may be absent; lslocks
            # is util-linux, which is essential, so one of the two always
            # answers. SIGTERM first, giving dpkg 10s to finish its current
            # package, and SIGKILL only what ignores it.
            (cls.KILL_LOCKS,
             f'LOCKS="{cls.LOCK_FILES}"; '
             'HOLDERS=$(sudo fuser $LOCKS 2>/dev/null '
             '| tr -s " " "\\n" | grep -E "^[0-9]+$" | sort -u); '
             'if [ -z "$HOLDERS" ]; then '
             'HOLDERS=$(sudo lslocks -n -o PID,PATH 2>/dev/null | tr -s " " '
             '| sed "s/^ //" | grep " /var/lib/dpkg\\| /var/lib/apt\\| /var/cache/apt" '
             '| cut -d" " -f1 | sort -u); fi; '
             'if [ -z "$HOLDERS" ]; then '
             'echo "no process is holding the apt locks"; exit 0; fi; '
             'echo "holders:" $HOLDERS; '
             'for p in $HOLDERS; do echo "  $p $(ps -o comm= -p $p 2>/dev/null)"; done; '
             'sudo kill -TERM $HOLDERS 2>/dev/null || true; '
             'sleep 10; '
             'SURVIVORS=$(for p in $HOLDERS; do sudo kill -0 $p 2>/dev/null '
             '&& echo $p; done); '
             'if [ -n "$SURVIVORS" ]; then echo "SIGKILL:" $SURVIVORS; '
             'sudo kill -KILL $SURVIVORS 2>/dev/null || true; sleep 2; fi; '
             'echo "killed:" $HOLDERS'),

            # A killed dpkg leaves half-configured packages that fail every
            # later install. Both commands are no-ops when nothing is broken.
            ('repair dpkg state',
             'sudo DEBIAN_FRONTEND=noninteractive dpkg --configure -a || true; '
             f'{cls.APT} -f install'),
        ]

    def install(self, step_timeout=0, stall_after=120):
        Print.info('Installing rust and cloning the repo...')
        # Each entry is (label, command) and runs as its own ssh command so the
        # progress view can name the step a host is sitting on. `source
        # $HOME/.cargo/env` used to be one link of a single `&&` chain; it is
        # now the per-step prelude below, since every step gets a fresh shell.
        # apt has to be usable before the first apt call, hence the unlock
        # steps up front.
        steps = self._apt_unlock_steps() + [
            ('apt-get update', f'{self.APT} update'),

            # build-essential prevents [error: linker `cc` not found]; clang is
            # missing from the Rocksdb installer. Installed in one apt call: it
            # takes the dpkg lock once instead of four times, and apt's own
            # output names the package being unpacked in the progress view.
            ('install build deps',
             f'{self.APT} install build-essential cmake libgmp-dev clang'),

            # Install rust (non-interactive).
            ('install rustup',
             'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y'),
            # Needs rustc >= 1.85 for edition-2024 transitive deps (time-macros >= 0.2.27,
            # pulled in by lambdaworks-math 0.13). The older 1.83.0 pin would silently
            # fail every cargo build. 1.88.0 is pinned rather than `stable` so benchmark
            # runs stay reproducible across the fleet.
            ('rustup install 1.88.0', 'rustup install 1.88.0'),
            ('rustup override 1.88.0', 'rustup override set 1.88.0'),

            # Clone the repo.
            ('clone/pull repo',
             f'(git clone {self.settings.repo_url} || (cd {self.settings.repo_name} ; git pull))'),
        ]
        hosts = self.manager.hosts(flat=True)
        runner = RemoteSteps(
            hosts,
            steps,
            connect_kwargs=self.connect,
            user='ubuntu',
            prelude='[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"',
            step_timeout=step_timeout,
            stall_after=stall_after,
            capture=(self.KILL_LOCKS,),
        )
        states = runner.run(title='Installing testbed')
        report_capture(states, self.KILL_LOCKS, header='apt lock holders:')
        failed = failures(states)
        if failed:
            report_failures(failed, len(hosts))
            raise BenchError(
                'Failed to install repo on testbed', StepError(failed)
            )
        Print.heading(f'Initialized testbed of {len(hosts)} nodes')

    def unlock_apt(self, step_timeout=0, stall_after=120):
        ''' Kill whatever holds the apt/dpkg locks, fleet-wide. '''
        Print.info('Clearing apt locks...')
        hosts = self.manager.hosts(flat=True)
        runner = RemoteSteps(
            hosts,
            self._apt_unlock_steps(),
            connect_kwargs=self.connect,
            user='ubuntu',
            step_timeout=step_timeout,
            stall_after=stall_after,
            capture=(self.KILL_LOCKS,),
        )
        states = runner.run(title='Unlocking apt')
        report_capture(states, self.KILL_LOCKS, header='apt lock holders:')
        failed = failures(states)
        if failed:
            report_failures(failed, len(hosts))
            raise BenchError('Failed to clear apt locks', StepError(failed))
        Print.heading(f'apt is unlocked on {len(hosts)} nodes')

    def kill(self, hosts=[], delete_logs=False):
        assert isinstance(hosts, list)
        assert isinstance(delete_logs, bool)
        hosts = hosts if hosts else self.manager.hosts(flat=True)
        delete_logs = CommandMaker.clean_logs() if delete_logs else 'true'
        cmd = [delete_logs, f'({CommandMaker.kill()} || true)']
        try:
            g = Group(*hosts, user='ubuntu', connect_kwargs=self.connect)
            g.run(' && '.join(cmd), hide=True)
        except GroupException as e:
            raise BenchError('Failed to kill nodes', FabricError(e))

    def _select_hosts(self, bench_parameters):
        # Collocate the primary and its workers on the same machine.
        nodes = max(bench_parameters.nodes)

        # Ensure there are enough hosts.
        hosts = self.manager.hosts()
        print("{} {}",sum(len(x) for x in hosts.values()), nodes)
        if sum(len(x) for x in hosts.values()) < nodes:
            return []

        # Select the hosts in different data centers.
        ordered = zip(*hosts.values())
        ordered = [x for y in ordered for x in y]
        return ordered[:nodes]


    def _background_run(self, host, command, log_file):
        name = splitext(basename(log_file))[0]
        cmd = f'tmux new -d -s "{name}" "{command} |& tee {log_file}"'
        c = Connection(host, user='ubuntu', connect_kwargs=self.connect)
        output = c.run(cmd, hide=True)
        self._check_stderr(output)

    def _update(self, hosts):
        ips = list(set(hosts))
        Print.info(
            f'Updating {len(ips)} machines (branch "{self.settings.branch}")...'
        )
        cmd = [
            f'(cd {self.settings.repo_name} && git fetch -f)',
            f'(cd {self.settings.repo_name} && git checkout -f {self.settings.branch})',
            f'(cd {self.settings.repo_name} && git pull -f)',
            'source $HOME/.cargo/env',
            # Defensive: every re-deploy re-asserts the 1.88.0 pin, overwriting any
            # leftover per-dir 1.83 pin (both at $HOME and inside the repo).
            # Hosts provisioned by the old install logic — or by the velox
            # README's manual setup — carry that pin until something overwrites it.
            'rustup install 1.88.0',
            'rustup override set 1.88.0',
            f'(cd {self.settings.repo_name} && rustup override set 1.88.0)',
            # libgmp3-dev needed by lambdaworks 0.13 transitive deps.
            'sudo apt-get install -y pkg-config libssl-dev libgmp3-dev',
            'export RUSTFLAGS="-C target-feature=+aes,+ssse3"',
            f'(cd {self.settings.repo_name} && {CommandMaker.compile()})',
            CommandMaker.alias_binaries(
                f'./{self.settings.repo_name}/target/release/'
            )
        ]
        g = Group(*ips, user='ubuntu', connect_kwargs=self.connect)
        print(g.run(' && '.join(cmd), hide=True))

    def _private_hosts(self, hosts):
        ''' Translate ssh (public) ips into the private ips the nodes dial.

        Falls back to the public ip for anything AWS reports no private address
        for -- the CloudLab/Chameleon path in InstanceManager.hosts() is one
        such case -- because a benchmark that runs over public ips is worth more
        than one that refuses to start. It says so loudly, since the fallback
        silently changes what the numbers mean.
        '''
        # WAN testbeds set "use_private_ips": false, because their regions each
        # have their own unpeered VPC and only the global ips are routable
        # between them.
        regions = self.settings.aws_regions
        if not self.settings.use_private_ips:
            Print.info(
                f'Addressing nodes by their global ips '
                f'("use_private_ips": false, {len(regions)} region(s))'
            )
            return list(hosts)

        # The same reasoning in reverse: private ips with more than one region
        # configured means the nodes will never reach each other.
        if len(regions) > 1:
            Print.warn(
                f'settings lists {len(regions)} regions ({", ".join(regions)}) '
                f'but "use_private_ips" is true. Private ips do not route '
                f'between regions without VPC peering or a transit gateway -- '
                f'the nodes will not reach each other. Set "use_private_ips": '
                f'false for a WAN testbed, or use a single region.'
            )

        private = self.manager.private_ips()
        missing = [x for x in hosts if x not in private]
        if missing:
            Print.warn(
                f'No private ip for {len(missing)}/{len(hosts)} hosts '
                f'({", ".join(missing[:5])}): the protocol will address these '
                f'over their public ips, so their traffic leaves the VPC.'
            )
        return [private.get(x, x) for x in hosts]

    def _config(self, hosts, bench_parameters):
        Print.info('Generating configuration files...')
        #print(hosts)
        # Cleanup all local configuration files.
        cmd = CommandMaker.cleanup()
        subprocess.run([cmd], shell=True, stderr=subprocess.DEVNULL)

        # Recompile the latest code.
        cmd = CommandMaker.compile().split()
        print("Running command ",cmd)
        subprocess.run(cmd, check=True, cwd=PathMaker.node_crate_path())

        # Create alias for the client and nodes binary.
        cmd = CommandMaker.alias_binaries(PathMaker.binary_path())
        subprocess.run([cmd], shell=True)

        # Generate the configuration files for Velox. The following code maps the ips of each party. 
        cmd = CommandMaker.generate_config_files(self.settings.base_port,self.settings.client_base_port,self.settings.client_run_port,len(hosts))
        subprocess.run(cmd,shell=True)
        # `hosts` holds public ips: that is what ssh below needs. The addresses
        # baked into ip_file/syncer/committee are what the nodes dial each
        # other on, so those use private ips and the n^2 protocol traffic stays
        # inside the VPC instead of hairpinning through the internet gateway.
        peers = self._private_hosts(hosts)

        names = [str(x) for x in range(len(hosts))]
        ip_file = ""
        syncer=""
        for x in range(len(hosts)):
            port = self.settings.base_port + x
            syncer_port = self.settings.client_base_port + x
            ip_file += peers[x]+ ":"+ str(port) + "\n"
            syncer += peers[x] + ":" + str(syncer_port) + "\n"
        ip_file += peers[0] + ":" + str(self.settings.client_run_port) + "\n"
        with open("ip_file", 'w') as f:
            f.write(ip_file)
        f.close()
        with open("syncer",'w') as f:
            f.write(syncer)
        f.close()
        #names = [str(x) for x in range(len(hosts))]
        addresses = OrderedDict(
            (x, [y] * (1 + 1)) for x, y in zip(names, peers)
        )
        # if bench_parameters.collocate:
        #     workers = bench_parameters.workers
        
        # else:
        
        committee = Committee(addresses, self.settings.base_port)
        committee.print(PathMaker.committee_file())

        # start the syncer on the first node first. 
        # Cleanup all nodes and upload configuration files.
        names = names[:len(names)-bench_parameters.faults]
        progress = progress_bar(names, prefix='Uploading config files:')
        for i,name in enumerate(progress):
            #for ip in committee.ips(name):
            c = Connection(hosts[i], user='ubuntu', connect_kwargs=self.connect)
            c.run(f'{CommandMaker.cleanup()} || true', hide=True)
            #c.put(PathMaker.committee_file(), '.')
            if i == 0:
                print('Node 0: writing syncer')
                c.put(PathMaker.syncer(),'.')
            # Write the configuration file to the remote machine. 
            c.put(PathMaker.key_file(i), '.')
            # Write the input file to the remote machine. 
            c.put(PathMaker.input_file(i),'.')
            # Write the ip_file to the remote machine. 
            c.put("ip_file",'.')
            
        Print.info('Booting primaries...')

        for i,ip in enumerate(hosts):
            #host = Committee.ip(address)
            if i == 0:
                # Run syncer first
                print('Running syncer')
                cmd = CommandMaker.run_syncer(
                    PathMaker.key_file(i),
                    bench_parameters.num_messages,
                    bench_parameters.compression_factor,
                    bench_parameters.num_batches
                )
                print('Running the following command on the remote machine:', cmd)
                log_file = PathMaker.syncer_log_file()
                self._background_run(ip, cmd, log_file)
            cmd = CommandMaker.run_primary(
                PathMaker.key_file(i),
                bench_parameters.num_messages,
                bench_parameters.compression_factor,
                bench_parameters.num_batches
            )
            log_file = PathMaker.primary_log_file(i)
            self._background_run(ip, cmd, log_file)
        return committee

    def _just_run(self, hosts, bench_parameters):
        # While calling this method, the configuration files have already been written to the remote machines. 
        Print.info('Booting primaries...')

        for i,ip in enumerate(hosts):
            if i == 0:
                # Run syncer first
                print('Running syncer')
                cmd = CommandMaker.run_syncer(
                    PathMaker.key_file(i),
                    bench_parameters.num_messages,
                    bench_parameters.compression_factor,
                    bench_parameters.num_batches
                )
                print(cmd)
                log_file = PathMaker.syncer_log_file()
                self._background_run(ip, cmd, log_file)
            cmd = CommandMaker.run_primary(
                PathMaker.key_file(i),
                bench_parameters.num_messages,
                bench_parameters.compression_factor,
                bench_parameters.num_batches
            )
            log_file = PathMaker.primary_log_file(i)
            self._background_run(ip, cmd, log_file)

    def _logs(self, hosts, faults, bench_parameters):
        # Delete local logs (if any).
        cmd = CommandMaker.clean_logs()
        subprocess.run([cmd], shell=True, stderr=subprocess.DEVNULL)

        # Download log files.
        #workers_addresses = committee.workers_addresses(faults)
        progress = progress_bar(hosts, prefix='Downloading workers logs:')
        for i, address in enumerate(progress):
            if i==0:
                c = Connection(address, user='ubuntu', connect_kwargs=self.connect)
                c.get(
                    PathMaker.syncer_log_file(),
                    local=PathMaker.syncer_local_log_file(
                        bench_parameters.nodes[0],
                        bench_parameters.num_messages,
                        bench_parameters.batch_size,
                        bench_parameters.compression_factor
                    )
                )
                c.get(
                    PathMaker.client_log_file(i), 
                    local= PathMaker.client_local_log_file(
                        i,
                        bench_parameters.nodes[0],
                        bench_parameters.num_messages,
                        bench_parameters.batch_size,
                        bench_parameters.compression_factor
                    )
                )

        Print.info('Downloaded logs from server, check latencies in the logs folder')

    def run(self, bench_parameters_dict, debug=False):
        assert isinstance(debug, bool)
        Print.heading('Starting remote benchmark')
        try:
            bench_parameters = BenchParameters(bench_parameters_dict)
        except ConfigError as e:
            raise BenchError('Invalid nodes or bench parameters', e)

        # Select which hosts to use.
        selected_hosts = self._select_hosts(bench_parameters)
        print("IP Addresses of selected machines: ", selected_hosts)
        if not selected_hosts:
            Print.warn('There are not enough instances available')
            return

        # Update nodes.
        try:
            self._update(selected_hosts)
        except (GroupException, ExecutionError) as e:
            e = FabricError(e) if isinstance(e, GroupException) else e
            raise BenchError('Failed to update nodes', e)

        # Upload all configuration files and run the protocol. 
        try:
            committee = self._config(
                selected_hosts, bench_parameters
            )
        except (subprocess.SubprocessError, GroupException) as e:
            e = FabricError(e) if isinstance(e, GroupException) else e
            raise BenchError('Failed to configure nodes', e)

    def justrun(self, bench_parameters_dict, debug=False):
        assert isinstance(debug, bool)
        Print.heading('Starting remote benchmark')
        try:
            bench_parameters = BenchParameters(bench_parameters_dict)
        except ConfigError as e:
            raise BenchError('Invalid nodes or bench parameters', e)

        # Select which hosts to use.
        selected_hosts = self._select_hosts(bench_parameters)
        print("IP addresses of selected hosts:", selected_hosts)
        if not selected_hosts:
            Print.warn('There are not enough instances available')
            return

        # Update nodes.
        try:
            self._update(selected_hosts)
        except (GroupException, ExecutionError) as e:
            e = FabricError(e) if isinstance(e, GroupException) else e
            raise BenchError('Failed to update nodes', e)

        # Upload all configuration files.
        try:
            committee = self._just_run(
                selected_hosts, bench_parameters
            )
        except (subprocess.SubprocessError, GroupException) as e:
            e = FabricError(e) if isinstance(e, GroupException) else e
            raise BenchError('Failed to configure nodes', e)

    def pull_logs(self, bench_parameters_dict, debug=False):
        assert isinstance(debug, bool)
        Print.heading('Starting remote benchmark')
        try:
            bench_parameters = BenchParameters(bench_parameters_dict)
        except ConfigError as e:
            raise BenchError('Invalid nodes or bench parameters', e)

        # Select which hosts to use.
        selected_hosts = self._select_hosts(bench_parameters)
        self._logs(selected_hosts,0, bench_parameters)
