import os
import logging
import sys

from enum import Enum

import toml

from codebuild_environment import get_codebuild_project_name, get_cloned_folder_path

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))
LOGGER.addHandler(logging.StreamHandler(sys.stderr))


def get_dlc_developer_config_path():
    dev_config_parent_dir = get_cloned_folder_path()
    return os.path.join(dev_config_parent_dir, "dlc_developer_config.toml")


config_data = None


def parse_dlc_developer_configs(section, option, tomlfile=get_dlc_developer_config_path()):
    global config_data

    if config_data is None:
        config_data = toml.load(tomlfile)

    return config_data.get(section, {}).get(option)


def get_buildspec_override():
    build_project_name = get_codebuild_project_name()
    return parse_dlc_developer_configs("buildspec_override", build_project_name)


def is_benchmark_mode_enabled():
    return parse_dlc_developer_configs("dev", "benchmark_mode")


def is_build_enabled():
    return parse_dlc_developer_configs("build", "do_build")


def is_ec2_test_enabled():
    return parse_dlc_developer_configs("test", "ec2_tests")


def is_ec2_efa_test_enabled():
    return parse_dlc_developer_configs("test", "ec2_efa_tests")


def is_ecs_test_enabled():
    return parse_dlc_developer_configs("test", "ecs_tests")


def is_eks_test_enabled():
    return parse_dlc_developer_configs("test", "eks_tests")


def is_sanity_test_enabled():
    return parse_dlc_developer_configs("test", "sanity_tests")


def is_sm_local_test_enabled():
    return parse_dlc_developer_configs("test", "sagemaker_local_tests")


def is_nightly_pr_test_mode_enabled():
    return parse_dlc_developer_configs("test", "nightly_pr_test_mode")


def is_scheduler_enabled():
    return parse_dlc_developer_configs("test", "use_scheduler")


def is_safety_check_test_enabled():
    return parse_dlc_developer_configs("test", "safety_check_test")


def is_ecr_scan_allowlist_feature_enabled():
    return parse_dlc_developer_configs("test", "ecr_scan_allowlist_feature")


class AllowedSMRemoteConfigValues(Enum):
    OFF = "off"
    RC = "rc"
    STANDARD = "standard"
    EFA = "efa"


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

    if sm_remote_tests_value in {
        True,
        AllowedSMRemoteConfigValues.STANDARD.value,
        AllowedSMRemoteConfigValues.RC.value,
        AllowedSMRemoteConfigValues.EFA.value,
    }:
        return True

    if sm_remote_tests_value != AllowedSMRemoteConfigValues.OFF.value:
        LOGGER.warning(
            f"Unrecognized sagemaker_remote_tests setting {sm_remote_tests_value}. "
            f"Please choose one of {allowed_values}. Disabling sagemaker remote tests."
        )
    return False


def are_sm_efa_tests_enabled():
    sm_remote_value = get_sagemaker_remote_tests_config_value()
    return sm_remote_value == AllowedSMRemoteConfigValues.EFA.value


def get_sagemaker_remote_efa_instance_type():
    """
    Get the config value for sagemaker_remote_efa_instance_type
    """
    return parse_dlc_developer_configs("test", "sagemaker_remote_efa_instance_type")
