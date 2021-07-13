import os
import re

import toml


def get_dlc_developer_config_path():
    root_dir_pattern = re.compile(r'^(\S+deep-learning-containers)')
    pwd = os.getcwd()
    dev_config_parent_dir = os.getenv("CODEBUILD_SRC_DIR")

    # Ensure we are inside some directory called "deep-learning-containers
    try:
        if not dev_config_parent_dir:
            dev_config_parent_dir = root_dir_pattern.match(pwd).group(1)
    except AttributeError as e:
        raise RuntimeError(f"Unable to find DLC root directory in path {pwd}, and no CODEBUILD_SRC_DIR set") from e

    return os.path.join(dev_config_parent_dir, "dlc_developer_config.toml")


def parse_dlc_developer_configs(section, option, tomlfile=get_dlc_developer_config_path()):
    data = toml.load(tomlfile)

    return data.get(section, {}).get(option)


def is_benchmark_mode_enabled():
    return parse_dlc_developer_configs("dev", "benchmark_mode")
