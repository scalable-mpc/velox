# Copyright(C) Facebook, Inc. and its affiliates.
'''Live progress reporting for long multi-step remote commands.

`Bench.install` used to send a single `&&`-joined shell command to every host
with `hide=True`: a step stuck on an apt lock looked exactly like a step that
was merely slow, because nothing at all was printed until the whole chain
finished on every machine. This module runs the same commands as *named steps*
and paints a live view of which step each host is on, how long it has been
there, and the last line it printed -- which is normally enough to see both
where and why a run is stuck.
'''
import shutil
import sys
import threading
from time import monotonic, sleep, strftime

from fabric import Connection

from benchmark.utils import Color


def format_duration(seconds):
    seconds = int(max(seconds, 0))
    if seconds >= 3600:
        return f'{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}'
    return f'{seconds // 60}:{seconds % 60:02d}'


class StepError(Exception):
    ''' Raised when at least one host failed to run a step. '''

    def __init__(self, failures):
        self.failures = failures
        hosts = ', '.join(sorted({x.host for x in failures})[:5])
        more = '' if len(failures) <= 5 else f' (+{len(failures) - 5} more)'
        step = failures[0].failed_step
        super().__init__(
            f'{len(failures)} host(s) failed at step "{step}": {hosts}{more}'
        )


class _Tail:
    ''' File-like sink keeping only the newest line a command printed.

    Output is watched rather than hidden so that a host waiting on an
    interactive prompt (apt config questions, needrestart) shows the prompt
    text instead of nothing. Prompts arrive without a trailing newline, so the
    unterminated tail counts as a line too.
    '''

    def __init__(self, state):
        self._state = state
        self._partial = ''

    def write(self, data):
        if not data:
            return
        self._partial += data
        *complete, self._partial = self._partial.split('\n')
        if len(self._partial) > 4096:  # a command printing one huge line
            self._partial = self._partial[-2048:]

        # Progress meters (apt, cargo, rustup) redraw with '\r'; only the last
        # segment of a line is on screen.
        lines = [x.split('\r')[-1].strip() for x in complete]
        lines = [x for x in lines if x]
        if not lines:
            tail = self._partial.split('\r')[-1].strip()
            lines = [tail] if tail else []
        self._state.note_output(lines[-1] if lines else '')

    def flush(self):
        pass


class _HostState:
    ''' Where a single host currently is, guarded for the renderer thread. '''

    CONNECTING, RUNNING, DONE, FAILED = 'connecting', 'running', 'done', 'failed'

    def __init__(self, host, total_steps):
        self.host = host
        self.total_steps = total_steps
        self._lock = threading.Lock()
        now = monotonic()
        self.status = self.CONNECTING
        self.step = 0               # 1-based; 0 means "still opening ssh"
        self.label = 'ssh connect'
        self.step_started = now
        self.last_output = now
        self.line = ''
        self.failed_step = None
        self.error = ''
        self.outputs = {}       # label -> full stdout, for captured steps

    def begin(self, step, label):
        with self._lock:
            self.status = self.RUNNING
            self.step = step
            self.label = label
            self.step_started = monotonic()
            self.last_output = self.step_started
            self.line = ''

    def record(self, label, text):
        with self._lock:
            self.outputs[label] = text

    def note_output(self, line):
        with self._lock:
            self.last_output = monotonic()
            if line:
                self.line = line

    def finish(self):
        with self._lock:
            self.status = self.DONE
            self.step = self.total_steps

    def fail(self, error):
        with self._lock:
            self.status = self.FAILED
            self.failed_step = self.label
            self.error = error

    def snapshot(self):
        with self._lock:
            now = monotonic()
            return {
                'host': self.host,
                'status': self.status,
                'step': self.step,
                'label': self.label,
                'elapsed': now - self.step_started,
                'quiet': now - self.last_output,
                'line': self.line,
                'failed_step': self.failed_step,
                'error': self.error,
                'outputs': dict(self.outputs),
                # A host that finished contributes every step; one still
                # connecting contributes none.
                'units': self.total_steps if self.status == self.DONE
                         else max(self.step - 1, 0),
            }


class _Renderer:
    ''' Paints the live block in place on a tty, or logs a line periodically. '''

    IN_FLIGHT_ROWS = 6
    HEIGHT = 5 + 2 * IN_FLIGHT_ROWS

    def __init__(self, title, states, total_steps, stall_after):
        self.title = title
        self.states = states
        self.total_steps = total_steps
        self.stall_after = stall_after
        self.started = monotonic()
        self.tty = sys.stdout.isatty()
        self._painted = False

    def _bar(self, fraction, length=30):
        filled = int(length * fraction)
        return '█' * filled + '-' * (length - filled)

    def _lines(self):
        snaps = [x.snapshot() for x in self.states]
        total_units = len(snaps) * self.total_steps
        units = sum(x['units'] for x in snaps)
        counts = {k: 0 for k in
                  (_HostState.CONNECTING, _HostState.RUNNING,
                   _HostState.DONE, _HostState.FAILED)}
        for x in snaps:
            counts[x['status']] += 1

        fraction = units / total_units if total_units else 1.0
        elapsed = monotonic() - self.started
        lines = [
            f'{self.title}: {len(snaps)} hosts x {self.total_steps} steps',
            f' |{self._bar(fraction)}| {100 * fraction:5.1f}%  '
            f'{units}/{total_units} steps  elapsed {format_duration(elapsed)}',
            f" hosts  {counts[_HostState.DONE]} done  "
            f"{counts[_HostState.RUNNING]} running  "
            f"{counts[_HostState.CONNECTING]} connecting  "
            f"{counts[_HostState.FAILED]} failed",
        ]

        in_flight = [x for x in snaps if x['status'] in
                     (_HostState.RUNNING, _HostState.CONNECTING)]
        in_flight.sort(key=lambda x: (x['step'], -x['elapsed']))
        if in_flight:
            slowest = in_flight[0]
            lines.append(
                f" furthest behind: step {slowest['step']}/{self.total_steps} "
                f"{slowest['label']}"
            )
            lines.append(' in flight (least progress first):')
        else:
            lines += ['', '']

        for snap in in_flight[:self.IN_FLIGHT_ROWS]:
            stalled = snap['quiet'] > self.stall_after
            quiet = f"  quiet for {format_duration(snap['quiet'])}" if stalled else ''
            head = (f"  {snap['host']:<16}[{snap['step']:>2}/{self.total_steps}] "
                    f"{snap['label']:<26} {format_duration(snap['elapsed'])}{quiet}")
            body = f"      | {snap['line'] or '(no output yet)'}"
            if stalled:
                head, body = f'{Color.WARNING}{head}', f'{body}{Color.END}'
            lines += [head, body]
        return lines

    def _paint(self):
        width = shutil.get_terminal_size((100, 24)).columns - 1
        lines = self._lines()
        lines += [''] * (self.HEIGHT - len(lines))
        out = sys.stdout
        if self._painted:
            out.write(f'\x1b[{self.HEIGHT}A')
        for line in lines[:self.HEIGHT]:
            # Truncate on the plain text so colour codes are never cut apart.
            if line.startswith(Color.WARNING):
                body = line[len(Color.WARNING):].rstrip(Color.END)
                line = f'{Color.WARNING}{body[:width]}{Color.END}'
            else:
                line = line[:width]
            out.write(f'\x1b[2K{line}\n')
        out.flush()
        self._painted = True

    def _log(self):
        snaps = [x.snapshot() for x in self.states]
        done = sum(1 for x in snaps if x['status'] == _HostState.DONE)
        stalled = [x for x in snaps
                   if x['status'] == _HostState.RUNNING
                   and x['quiet'] > self.stall_after]
        note = ''
        if stalled:
            worst = max(stalled, key=lambda x: x['quiet'])
            note = (f" | slowest {worst['host']} step {worst['step']} "
                    f"{worst['label']} quiet {format_duration(worst['quiet'])}: "
                    f"{worst['line']}")
        print(f"[{strftime('%H:%M:%S')}] {self.title}: {done}/{len(snaps)} "
              f'hosts done{note}', flush=True)

    def refresh(self):
        if self.tty:
            self._paint()
        else:
            self._log()

    def finalize(self):
        ''' Leave the finished block on screen and move the cursor below it. '''
        if self.tty and self._painted:
            self._paint()
        else:
            self._log()


class RemoteSteps:
    ''' Runs an ordered list of named shell steps on every host, in parallel.

    Steps run as separate ssh commands rather than one `&&` chain so that each
    one can be timed and reported. Anything a step needs in its environment
    must therefore come from `prelude`, which is prepended to every command.
    '''

    def __init__(self, hosts, steps, connect_kwargs, user='ubuntu', prelude='',
                 connect_timeout=30, step_timeout=0, stall_after=120,
                 capture=()):
        assert steps, 'no steps to run'
        self.hosts = list(hosts)
        self.steps = list(steps)
        self.connect_kwargs = connect_kwargs
        self.user = user
        self.prelude = prelude
        self.connect_timeout = connect_timeout
        self.step_timeout = step_timeout or None
        self.stall_after = stall_after
        # Labels whose full stdout is kept for `report_capture`. Only worth it
        # for steps that report a finding; the live view keeps one line.
        self.capture = set(capture)

    def _run_host(self, state, cancel):
        c = None
        try:
            c = Connection(
                state.host, user=self.user, connect_kwargs=self.connect_kwargs,
                connect_timeout=self.connect_timeout
            )
            c.open()
            for i, (label, command) in enumerate(self.steps, start=1):
                if cancel.is_set():
                    return
                state.begin(i, label)
                sink = _Tail(state)
                if self.prelude:
                    command = f'{self.prelude}; {command}'
                result = c.run(
                    command,
                    warn=True,          # collect the failure, don't raise here
                    pty=False,
                    in_stream=False,    # never let a remote prompt eat our stdin
                    out_stream=sink,
                    err_stream=sink,
                    timeout=self.step_timeout,
                )
                if label in self.capture:
                    state.record(label, result.stdout)
                if result.exited != 0:
                    tail = (result.stderr or result.stdout or '').strip()
                    state.fail(
                        f'exit code {result.exited}\n'
                        + '\n'.join(tail.splitlines()[-15:])
                    )
                    return
            state.finish()
        except Exception as e:  # ssh failure, timeout, unreachable host, ...
            state.fail(f'{type(e).__name__}: {e}')
        finally:
            if c is not None:
                try:
                    c.close()
                except Exception:
                    pass

    def run(self, title='Running', refresh=0.4):
        ''' Run every step on every host, returning every host's final state. '''
        states = [_HostState(h, len(self.steps)) for h in self.hosts]
        renderer = _Renderer(title, states, len(self.steps), self.stall_after)
        cancel = threading.Event()

        # Daemon threads (rather than a pool) so that a Ctrl-C on a wedged ssh
        # command returns control immediately instead of blocking on a join.
        threads = [
            threading.Thread(
                target=self._run_host, args=(s, cancel), daemon=True
            ) for s in states
        ]
        for t in threads:
            t.start()

        tick = refresh if renderer.tty else 15
        try:
            while any(t.is_alive() for t in threads):
                renderer.refresh()
                sleep(tick)
            renderer.finalize()
        except KeyboardInterrupt:
            cancel.set()
            renderer.finalize()
            self._report_interrupt(states)
            raise

        return states

    def _report_interrupt(self, states):
        snaps = [x.snapshot() for x in states
                 if x.snapshot()['status'] in
                 (_HostState.RUNNING, _HostState.CONNECTING)]
        print(f'\n{Color.WARNING}Interrupted with {len(snaps)} host(s) still '
              f'in flight:{Color.END}')
        for snap in sorted(snaps, key=lambda x: (x['step'], -x['elapsed'])):
            print(f"  {snap['host']:<16}[{snap['step']:>2}/{len(self.steps)}] "
                  f"{snap['label']:<26} {format_duration(snap['elapsed'])} "
                  f"(quiet {format_duration(snap['quiet'])})")
            print(f"      | {snap['line'] or '(no output yet)'}")


def failures(states):
    ''' The states of the hosts that did not finish every step. '''
    return [s for s in states if s.snapshot()['status'] != _HostState.DONE]


def _group_hosts(pairs):
    ''' Group (host, text) pairs by identical text, biggest group first. '''
    groups = {}
    for host, text in pairs:
        groups.setdefault(text, []).append(host)
    return sorted(groups.items(), key=lambda kv: -len(kv[1]))


def _host_list(hosts, limit=5):
    shown = ', '.join(hosts[:limit])
    return shown if len(hosts) <= limit else f'{shown} (+{len(hosts) - limit} more)'


def report_capture(states, label, header=None):
    ''' Print what a captured step found, grouping hosts that reported the same.

    The live view only ever shows a step's newest line; this is how a step that
    reports a finding (rather than just making progress) gets read.
    '''
    pairs = [(s.snapshot()['host'],
              (s.snapshot()['outputs'].get(label) or '').strip())
             for s in states]
    print(f'\n{Color.BOLD}{header or label}{Color.END}')
    for text, hosts in _group_hosts(pairs):
        print(f'\n{len(hosts)} host(s): {_host_list(hosts)}')
        for line in (text or '(step did not run)').splitlines():
            print(f'  {line}')


def report_failures(failed, total_hosts):
    ''' Print why hosts failed, grouping hosts that failed the same way. '''
    print(f'\n{Color.BOLD}{Color.FAIL}{len(failed)}/{total_hosts} hosts '
          f'failed{Color.END}')
    pairs = [(s.snapshot()['host'],
              (s.snapshot()['failed_step'] or s.snapshot()['label'],
               s.snapshot()['error']))
             for s in failed]
    for (step, error), hosts in _group_hosts(pairs):
        print(f'\n{Color.FAIL}step "{step}" failed on {len(hosts)} host(s): '
              f'{_host_list(hosts)}{Color.END}')
        for line in (error or '').splitlines():
            print(f'  {line}')
