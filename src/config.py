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


def is_build_enabled():
    return parse_dlc_developer_configs("build", "do_build")


def is_ec2_test_enabled():
    return parse_dlc_developer_configs("test", "ec2_tests")


def is_ec2_benchmark_test_enabled():
    return parse_dlc_developer_configs("test", "ec2_benchmark_tests")


def are_heavy_instance_ec2_tests_enabled():
    return parse_dlc_developer_configs("test", "ec2_tests_on_heavy_instances")


def is_ecs_test_enabled():
    return parse_dlc_developer_configs("test", "ecs_tests")


def is_eks_test_enabled():
    return parse_dlc_developer_configs("test", "eks_tests")


def is_sm_remote_test_enabled():
    return parse_dlc_developer_configs("test", "sagemaker_remote_tests")


def is_sm_rc_test_enabled():
    return parse_dlc_developer_configs("test", "sagemaker_rc_tests")


def is_sm_efa_test_enabled():
    return parse_dlc_developer_configs("test", "sagemaker_efa_tests")


def is_sm_benchmark_test_enabled():
    return parse_dlc_developer_configs("test", "sagemaker_benchmark_tests")


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


def get_sagemaker_remote_efa_instance_type():
    """
    Get the config value for sagemaker_remote_efa_instance_type
    """
    return parse_dlc_developer_configs("test", "sagemaker_remote_efa_instance_type")
