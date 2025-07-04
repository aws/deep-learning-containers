import logging
import os
import sys
from enum import Enum

import toml

from codebuild_environment import get_cloned_folder_path, get_codebuild_project_name

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


def is_deep_canary_mode_enabled():
    return parse_dlc_developer_configs("dev", "deep_canary_mode")


def is_graviton_mode_enabled():
    return parse_dlc_developer_configs("dev", "graviton_mode")


def is_arm64_mode_enabled():
    return parse_dlc_developer_configs("dev", "arm64_mode")


def is_build_enabled():
    return parse_dlc_developer_configs("build", "do_build")


def is_autopatch_build_enabled(buildspec_path=None):
    from buildspec import Buildspec

    if not buildspec_path:
        return False
    image_buildspec_object = Buildspec()
    image_buildspec_object.load(buildspec_path)
    autopatch_build_flag = image_buildspec_object.get("autopatch_build", "False").lower() == "true"
    return autopatch_build_flag


def is_ec2_test_enabled():
    return parse_dlc_developer_configs("test", "ec2_tests")


def is_ec2_benchmark_test_enabled():
    return parse_dlc_developer_configs("test", "ec2_benchmark_tests")


def are_heavy_instance_ec2_tests_enabled():
    return parse_dlc_developer_configs("test", "ec2_tests_on_heavy_instances")


def is_ipv6_test_enabled():
    return parse_dlc_developer_configs("test", "enable_ipv6")


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


def is_security_test_enabled():
    return parse_dlc_developer_configs("test", "security_tests")


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


def is_notify_test_failures_enabled():
    return parse_dlc_developer_configs("notify", "notify_test_failures")


class AllowedNotificationSeverity(Enum):
    MEDIUM = "medium"
    HIGH = "high"


def get_notification_severity():
    notification_severity = parse_dlc_developer_configs("notify", "notification_severity")
    allowed_values = [cfg_opt.value for cfg_opt in AllowedNotificationSeverity]

    if isinstance(notification_severity, str):
        notification_severity = notification_severity.lower().strip()

    if notification_severity in {
        AllowedNotificationSeverity.HIGH.value,
        AllowedNotificationSeverity.MEDIUM.value,
    }:
        return notification_severity

    if notification_severity != "":
        LOGGER.warning(
            f"Unrecognized notification_severity setting {notification_severity}. "
            f"Please choose one of {allowed_values}. Using medium severity for notification"
        )

    return AllowedNotificationSeverity.MEDIUM.value


def get_ipv6_vpc_name():
    """
    Get the config value for ipv6_vpc_name
    """
    return parse_dlc_developer_configs("test", "ipv6_vpc_name")


def get_sagemaker_remote_efa_instance_type():
    """
    Get the config value for sagemaker_remote_efa_instance_type
    """
    return parse_dlc_developer_configs("test", "sagemaker_remote_efa_instance_type")


def is_pr_build_job_flavor_dedicated():
    """
    Return true if the build job is dedicated to any flavor of image
    :return: bool True/False
    """
    build_job_is_ei_dedicated = os.getenv("EIA_DEDICATED", "false").lower() == "true"
    build_job_is_neuron_dedicated = os.getenv("NEURON_DEDICATED", "false").lower() == "true"
    build_job_is_neuronx_dedicated = os.getenv("NEURONX_DEDICATED", "false").lower() == "true"
    build_job_is_graviton_dedicated = os.getenv("GRAVITON_DEDICATED", "false").lower() == "true"
    build_job_is_arm64_dedicated = os.getenv("ARM64_DEDICATED", "false").lower() == "true"
    build_job_is_habana_dedicated = os.getenv("HABANA_DEDICATED", "false").lower() == "true"
    build_job_is_hf_trcomp_dedicated = (
        os.getenv("HUGGINFACE_TRCOMP_DEDICATED", "false").lower() == "true"
    )
    build_job_is_trcomp_dedicated = os.getenv("TRCOMP_DEDICATED", "false").lower() == "true"

    return (
        build_job_is_ei_dedicated
        or build_job_is_neuron_dedicated
        or build_job_is_neuronx_dedicated
        or build_job_is_graviton_dedicated
        or build_job_is_arm64_dedicated
        or build_job_is_habana_dedicated
        or build_job_is_hf_trcomp_dedicated
        or build_job_is_trcomp_dedicated
    )


def does_dev_config_enable_any_build_modes():
    """
    Return True if the dev config file enables any specific build mode
    :return: bool True/False
    """
    dev_config_enables_ei_build_mode = parse_dlc_developer_configs("dev", "ei_mode")
    dev_config_enables_neuron_build_mode = parse_dlc_developer_configs("dev", "neuron_mode")
    dev_config_enables_neuronx_build_mode = parse_dlc_developer_configs("dev", "neuronx_mode")
    dev_config_enables_graviton_build_mode = parse_dlc_developer_configs("dev", "graviton_mode")
    dev_config_enables_arm64_build_mode = parse_dlc_developer_configs("dev", "arm64_mode")
    dev_config_enables_habana_build_mode = parse_dlc_developer_configs("dev", "habana_mode")
    dev_config_enables_hf_trcomp_build_mode = parse_dlc_developer_configs("dev", "hf_trcomp_mode")
    dev_config_enables_trcomp_build_mode = parse_dlc_developer_configs("dev", "trcomp_mode")

    return (
        dev_config_enables_ei_build_mode
        or dev_config_enables_neuron_build_mode
        or dev_config_enables_neuronx_build_mode
        or dev_config_enables_graviton_build_mode
        or dev_config_enables_arm64_build_mode
        or dev_config_enables_habana_build_mode
        or dev_config_enables_hf_trcomp_build_mode
        or dev_config_enables_trcomp_build_mode
    )


def is_training_or_inference_enabled_for_this_pr_build():
    """
    Return True if training/inference image types are enabled on PR, and if this PR job is dedicated
    to building training/inference image types, respectively.
    Alternatively, if PR job is not dedicated to building one type of image, then return True.
    :return: bool True/False
    """
    image_type = os.getenv("IMAGE_TYPE", "").lower()

    training_dedicated = image_type == "training"
    training_enabled = parse_dlc_developer_configs("build", "build_training")

    inference_dedicated = image_type == "inference"
    inference_enabled = parse_dlc_developer_configs("build", "build_inference")

    return (
        (training_dedicated and training_enabled)
        or (inference_dedicated and inference_enabled)
        or (image_type == "")
    )


def is_framework_enabled_for_this_pr_build(framework):
    """
    Return True if the framework is enabled for this PR build job.
    :param framework: str Framework name
    :return: bool True/False
    """
    frameworks_to_build = parse_dlc_developer_configs("build", "build_frameworks")
    return framework in frameworks_to_build


def is_ei_builder_enabled_for_this_pr_build(framework):
    """
    Return True if this PR job should build EI DLCs for the given framework name.
    :param framework: str Framework name
    :return: bool True/False
    """
    build_job_is_ei_dedicated = os.getenv("EIA_DEDICATED", "false").lower() == "true"
    dev_config_enables_ei_build_mode = parse_dlc_developer_configs("dev", "ei_mode")
    return (
        build_job_is_ei_dedicated
        and dev_config_enables_ei_build_mode
        and is_framework_enabled_for_this_pr_build(framework)
        and is_training_or_inference_enabled_for_this_pr_build()
    )


def is_neuron_builder_enabled_for_this_pr_build(framework):
    """
    Return True if this PR job should build Neuron DLCs for the given framework name.
    :param framework: str Framework name
    :return: bool True/False
    """
    build_job_is_neuron_dedicated = os.getenv("NEURON_DEDICATED", "false").lower() == "true"
    dev_config_enables_neuron_build_mode = parse_dlc_developer_configs("dev", "neuron_mode")
    return (
        build_job_is_neuron_dedicated
        and dev_config_enables_neuron_build_mode
        and is_framework_enabled_for_this_pr_build(framework)
        and is_training_or_inference_enabled_for_this_pr_build()
    )


def is_neuronx_builder_enabled_for_this_pr_build(framework):
    """
    Return True if this PR job should build Neuronx DLCs for the given framework name.
    :param framework: str Framework name
    :return: bool True/False
    """
    build_job_is_neuronx_dedicated = os.getenv("NEURONX_DEDICATED", "false").lower() == "true"
    dev_config_enables_neuronx_build_mode = parse_dlc_developer_configs("dev", "neuronx_mode")
    return (
        build_job_is_neuronx_dedicated
        and dev_config_enables_neuronx_build_mode
        and is_framework_enabled_for_this_pr_build(framework)
        and is_training_or_inference_enabled_for_this_pr_build()
    )


def is_graviton_builder_enabled_for_this_pr_build(framework):
    """
    Return True if this PR job should build Graviton DLCs for the given framework name.
    :param framework: str Framework name
    :return: bool True/False
    """
    build_job_is_graviton_dedicated = os.getenv("GRAVITON_DEDICATED", "false").lower() == "true"
    dev_config_enables_graviton_build_mode = is_graviton_mode_enabled()
    return (
        build_job_is_graviton_dedicated
        and dev_config_enables_graviton_build_mode
        and is_framework_enabled_for_this_pr_build(framework)
        and is_training_or_inference_enabled_for_this_pr_build()
    )


def is_arm64_builder_enabled_for_this_pr_build(framework):
    """
    Return True if this PR job should build ARM64 DLCs for the given framework name.
    :param framework: str Framework name
    :return: bool True/False
    """
    build_job_is_arm64_dedicated = os.getenv("ARM64_DEDICATED", "false").lower() == "true"
    dev_config_enables_arm64_build_mode = is_arm64_mode_enabled()
    return (
        build_job_is_arm64_dedicated
        and dev_config_enables_arm64_build_mode
        and is_framework_enabled_for_this_pr_build(framework)
        and is_training_or_inference_enabled_for_this_pr_build()
    )


def is_habana_builder_enabled_for_this_pr_build(framework):
    """
     Return True if this PR job should build Habana DLCs for the given framework name.
    :param framework: str Framework name
    :return: bool True/False
    """
    build_job_is_habana_dedicated = os.getenv("HABANA_DEDICATED", "false").lower() == "true"
    dev_config_enables_habana_build_mode = parse_dlc_developer_configs("dev", "habana_mode")
    return (
        build_job_is_habana_dedicated
        and dev_config_enables_habana_build_mode
        and is_framework_enabled_for_this_pr_build(framework)
        and is_training_or_inference_enabled_for_this_pr_build()
    )


def is_hf_trcomp_builder_enabled_for_this_pr_build(framework):
    """
    Return True if this PR job should build HF Training Compiler DLCs for the given framework name.
    :param framework: str Framework name
    :return: bool True/False
    """
    build_job_is_trcomp_dedicated = (
        os.getenv("HUGGINFACE_TRCOMP_DEDICATED", "false").lower() == "true"
    )
    dev_config_enables_trcomp_build_mode = parse_dlc_developer_configs(
        "dev", "huggingface_trcomp_mode"
    )
    return (
        build_job_is_trcomp_dedicated
        and dev_config_enables_trcomp_build_mode
        and is_framework_enabled_for_this_pr_build(framework)
        and is_training_or_inference_enabled_for_this_pr_build()
    )


def is_trcomp_builder_enabled_for_this_pr_build(framework):
    """
    Return True if this PR job should build Training Compiler DLCs for the given framework name.
    :param framework: str Framework name
    :return: bool True/False
    """
    build_job_is_trcomp_dedicated = os.getenv("TRCOMP_DEDICATED", "false").lower() == "true"
    dev_config_enables_trcomp_build_mode = parse_dlc_developer_configs("dev", "trcomp_mode")
    return (
        build_job_is_trcomp_dedicated
        and dev_config_enables_trcomp_build_mode
        and is_framework_enabled_for_this_pr_build(framework)
        and is_training_or_inference_enabled_for_this_pr_build()
    )


def is_general_builder_enabled_for_this_pr_build(framework):
    """
    Return True if this PR job should build standard DLCs that do not belong to any special flavor,
    for the given framework name.
    :param framework: str Framework name
    :return: bool True/False
    """
    build_job_is_generic = not is_pr_build_job_flavor_dedicated()
    dev_config_is_generic_build_mode = not does_dev_config_enable_any_build_modes()
    return (
        build_job_is_generic
        and dev_config_is_generic_build_mode
        and is_framework_enabled_for_this_pr_build(framework)
        and is_training_or_inference_enabled_for_this_pr_build()
    )
