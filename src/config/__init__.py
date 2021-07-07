import os

import toml

from invoke.context import Context


def get_dlc_developer_config_path():
    ctx = Context()
    alt_root_dir = ctx.run("git rev-parse --show-toplevel", hide=True).stdout.strip()

    return os.path.join(os.getenv("CODEBUILD_SRC_DIR", alt_root_dir), "dlc_developer_config.toml")


def parse_dlc_developer_configs(section, option, tomlfile=get_dlc_developer_config_path()):
    data = toml.load(tomlfile)

    return data.get(section, {}).get(option)
