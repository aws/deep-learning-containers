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
    parse_dlc_developer_configs("test", "efa_tests")


def is_scheduler_enabled():
    parse_dlc_developer_configs("test", "use_scheduler")


def get_allowed_sagemaker_remote_tests_config_values(disabled_only=False, enabled_only=False):
    """
    Retrieve allowed SM remote tests config values
    """
    disabled = ("off", "", False)
    enabled = ("default", "release_candidate")
    if disabled_only:
        return disabled
    if enabled_only:
        return enabled
    return enabled + disabled


def get_sagemaker_remote_tests_config_value():
    sm_config_value = parse_dlc_developer_configs("test", "sagemaker_remote_tests")
    allowed_config_values = get_allowed_sagemaker_remote_tests_config_values()
    if sm_config_value not in allowed_config_values:
        LOGGER.warning(
            f"Unrecognized sagemaker_remote_tests config {sm_config_value}. "
            f"Please ensure it is one of {allowed_config_values}, or your tests may not get triggered as expected."
        )
    return sm_config_value


def is_sm_remote_test_enabled():
    sm_remote_tests_value = get_sagemaker_remote_tests_config_value()
    # disable SM tests if the sm_remote_tests_value is one of our 'disable' values
    if sm_remote_tests_value in get_allowed_sagemaker_remote_tests_config_values(disabled_only=True):
        return False
    # enable SM tests if the sm_remote_tests_value is one of our 'enable' values
    elif sm_remote_tests_value in get_allowed_sagemaker_remote_tests_config_values(enabled_only=True):
        return True
    # disable SM tests if bool(sm_remote_tests_value) == False. Warn in this case.
    elif not sm_remote_tests_value:
        LOGGER.warning(
            f"Disabling the SM remote tests, but unrecognized `False` boolean {sm_remote_tests_value} "
            f"detected for sagemaker_remote_tests config."
        )
        return False

    # enable SM tests if bool(sm_remote_tests_value) == True and the value is unrecognized. Warn in this case.
    LOGGER.warning(
        f"Enabling the SM remote tests, but unrecognized `True` boolean {sm_remote_tests_value} "
        f"detected for sagemaker_remote_tests config. Behavior may not be as expected."
    )
    return True
