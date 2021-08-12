import os
import re
import logging
import sys

import toml

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))
LOGGER.addHandler(logging.StreamHandler(sys.stderr))


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


def is_build_enabled():
    return parse_dlc_developer_configs("build", "do_build")


def get_allowed_sagemaker_remote_tests_config_values():
    """
    Retrieve allowed SM remote tests config values
    """
    return "", "default", "release_candidate"


def get_sagemaker_remote_tests_config_value():
    sm_config_value = parse_dlc_developer_configs("test", "sagemaker_remote_tests")
    allowed_config_values = get_allowed_sagemaker_remote_tests_config_values()
    if sm_config_value not in allowed_config_values:
        LOGGER.warning(
            f"Unrecognized sagemaker_remote_tests config {sm_config_value}. "
            f"Please ensure it is one of {allowed_config_values}, or your tests may not get triggered as expected."
        )
    return sm_config_value
