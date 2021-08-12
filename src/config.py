import os
import re
import logging
import sys

from enum import Enum

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


def is_ec2_test_enabled():
    return parse_dlc_developer_configs("test", "ec2_tests")


def is_ecs_test_enabled():
    return parse_dlc_developer_configs("test", "ecs_tests")


def is_eks_test_enabled():
    return parse_dlc_developer_configs("test", "eks_tests")


def is_sanity_test_enabled():
    return parse_dlc_developer_configs("test", "sanity_tests")


def is_sm_local_test_enabled():
    return parse_dlc_developer_configs("test", "sagemaker_local_tests")


def are_efa_tests_enabled():
    return parse_dlc_developer_configs("test", "efa_tests")


def is_scheduler_enabled():
    return parse_dlc_developer_configs("test", "use_scheduler")


class AllowedSMRemoteConfigValues(Enum):
    OFF = "off"
    RC = "rc"
    DEFAULT = "default"


def get_sagemaker_remote_tests_config_value():
    """
    Get the actual config option for sm remote tests
    """
    return parse_dlc_developer_configs("test", "sagemaker_remote_tests")


def is_sm_remote_test_enabled():
    """
    Check to see if sm remote test is enabled by config
    """
    sm_remote_tests_value = get_sagemaker_remote_tests_config_value()
    allowed_values = [cfg_opt.value for cfg_opt in AllowedSMRemoteConfigValues]

    # Sanitize value in case of extra whitespace or inconsistent capitalization
    if isinstance(sm_remote_tests_value, str):
        sm_remote_tests_value = sm_remote_tests_value.lower().strip()

    if sm_remote_tests_value == AllowedSMRemoteConfigValues.OFF.value:
        return False
    # Support "true" so as not to break existing workflows
    if sm_remote_tests_value is True:
        return True
    if sm_remote_tests_value in (AllowedSMRemoteConfigValues.DEFAULT.value, AllowedSMRemoteConfigValues.RC.value):
        return True
    LOGGER.warning(
        f"Unrecognized sagemaker_remote_tests setting {sm_remote_tests_value}. Please choose one of {allowed_values}. "
        f"Disabling sagemaker remote tests."
    )
    return False
