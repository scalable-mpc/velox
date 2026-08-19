# Copyright(C) Facebook, Inc. and its affiliates.
from json import load, JSONDecodeError


class SettingsError(Exception):
    pass


class Settings:
    def __init__(self, key_name, key_path, base_port,client_base_port,client_run_port, repo_name, repo_url,
                 branch, instance_type, aws_regions, use_private_ips=True):
        inputs_str = [
            key_name, key_path, repo_name, repo_url, branch, instance_type
        ]
        if isinstance(aws_regions, list):
            regions = aws_regions
        else:
            regions = [aws_regions]
        inputs_str += regions
        ok = all(isinstance(x, str) for x in inputs_str)
        ok &= isinstance(base_port, int)
        ok &= isinstance(use_private_ips, bool)
        ok &= len(regions) > 0
        if not ok:
            raise SettingsError('Invalid settings types')

        self.key_name = key_name
        self.key_path = key_path

        self.base_port = base_port

        self.client_base_port = client_base_port
        self.client_run_port = client_run_port

        self.repo_name = repo_name
        self.repo_url = repo_url
        self.branch = branch

        self.instance_type = instance_type
        self.aws_regions = regions

        # Whether the protocol addresses nodes by their private ip. True keeps
        # the n^2 traffic inside the VPC, which is what a single-region testbed
        # wants. A WAN testbed spans regions whose VPCs are not peered, so its
        # nodes can only reach each other over their global (public) ips.
        self.use_private_ips = use_private_ips

    @classmethod
    def load(cls, filename):
        try:
            with open(filename, 'r') as f:
                data = load(f)

            return cls(
                data['key']['name'],
                data['key']['path'],
                data['port'],
                data['client_base_port'],
                data['client_run_port'],
                data['repo']['name'],
                data['repo']['url'],
                data['repo']['branch'],
                data['instances']['type'],
                data['instances']['regions'],
                # Optional so that older settings files keep working; they are
                # single-region, where private ips are the right default.
                data.get('use_private_ips', True),
            )
        except (OSError, JSONDecodeError) as e:
            raise SettingsError(str(e))

        except KeyError as e:
            raise SettingsError(f'Malformed settings: missing key {e}')
