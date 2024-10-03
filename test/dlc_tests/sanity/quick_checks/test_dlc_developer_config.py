import pytest

from src import config


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("dlc_developer_config")
def test_developer_configuration():
    """
    Ensure that defaults are set back to normal before merge
    """
    # Check dev settings
    assert config.parse_dlc_developer_configs("dev", "partner_developer") == ""
    assert config.parse_dlc_developer_configs("dev", "ei_mode") is False
    assert config.parse_dlc_developer_configs("dev", "neuron_mode") is False
    assert config.parse_dlc_developer_configs("dev", "neuronx_mode") is False
    assert config.parse_dlc_developer_configs("dev", "graviton_mode") is False
    assert config.parse_dlc_developer_configs("dev", "habana_mode") is False
    assert config.parse_dlc_developer_configs("dev", "trcomp_mode") is False
    assert config.parse_dlc_developer_configs("dev", "deep_canary_mode") is False

    assert config.is_deep_canary_mode_enabled() is False
    assert config.is_deep_canary_mode_enabled() is False

    # Check build settings
    assert config.parse_dlc_developer_configs("build", "build_frameworks") == []
    assert config.parse_dlc_developer_configs("build", "build_training") is True
    assert config.parse_dlc_developer_configs("build", "build_inference") is True
    assert config.parse_dlc_developer_configs("build", "do_build") is True

    # Check test settings
    assert config.parse_dlc_developer_configs("test", "sanity_tests") is True
    assert config.parse_dlc_developer_configs("test", "sagemaker_remote_tests") is True
    assert config.parse_dlc_developer_configs("test", "sagemaker_efa_tests") is False
    assert config.parse_dlc_developer_configs("test", "sagemaker_rc_tests") is False
    assert config.parse_dlc_developer_configs("test", "sagemaker_remote_efa_instance_type") == ""
    assert config.parse_dlc_developer_configs("test", "sagemaker_local_tests") is True
    assert config.parse_dlc_developer_configs("test", "ecs_tests") is True
    assert config.parse_dlc_developer_configs("test", "eks_tests") is True
    assert config.parse_dlc_developer_configs("test", "ec2_tests") is True
    assert config.parse_dlc_developer_configs("test", "ec2_benchmark_tests") is False
    assert config.parse_dlc_developer_configs("test", "sagemaker_benchmark_tests") is False
    assert config.parse_dlc_developer_configs("test", "ec2_tests_on_heavy_instances") is False
    assert config.parse_dlc_developer_configs("test", "nightly_pr_test_mode") is False
    assert config.parse_dlc_developer_configs("test", "use_scheduler") is False
    assert config.parse_dlc_developer_configs("test", "safety_check_test") is False
    assert config.parse_dlc_developer_configs("test", "ecr_scan_allowlist_feature") is False

    # Check notify settings
    assert config.parse_dlc_developer_configs("notify", "notify_test_failures") is False
    assert config.parse_dlc_developer_configs("notify", "notification_severity") == "medium"


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("dlc_developer_config")
def test_developer_config_wrappers_defaults():
    """
    Test defaults of config file wrappers
    """
    # Check test settings
    assert config.is_sanity_test_enabled() is True
    assert config.is_sm_local_test_enabled() is True
    assert config.is_sm_remote_test_enabled() is True
    assert config.is_sm_efa_test_enabled() is False
    assert config.is_sm_rc_test_enabled() is False
    assert config.get_sagemaker_remote_efa_instance_type() == ""
    assert config.is_ecs_test_enabled() is True
    assert config.is_eks_test_enabled() is True
    assert config.is_ec2_test_enabled() is True
    assert config.is_ec2_benchmark_test_enabled() is False
    assert config.is_sm_benchmark_test_enabled() is False
    assert config.are_heavy_instance_ec2_tests_enabled() is False
    assert config.is_nightly_pr_test_mode_enabled() is False
    assert config.is_scheduler_enabled() is False
    assert config.is_safety_check_test_enabled() is False
    assert config.is_ecr_scan_allowlist_feature_enabled() is False
    assert config.is_notify_test_failures_enabled() is False
    assert config.get_notification_severity() == "medium"


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("dlc_build_version_override_config")
def test_build_version_override_configuration():
    """
    Ensure that buildspec override defaults are set back to normal before merge
    """
    assert config.parse_dlc_developer_configs("buildspec_override", "dlc-pr-mxnet-training") == ""
    assert config.parse_dlc_developer_configs("buildspec_override", "dlc-pr-pytorch-training") == ""
    assert (
        config.parse_dlc_developer_configs("buildspec_override", "dlc-pr-tensorflow-2-training")
        == ""
    )
    assert (
        config.parse_dlc_developer_configs("buildspec_override", "dlc-pr-autogluon-training") == ""
    )

    assert (
        config.parse_dlc_developer_configs(
            "buildspec_override", "dlc-pr-huggingface-tensorflow-training"
        )
        == ""
    )
    assert (
        config.parse_dlc_developer_configs(
            "buildspec_override", "dlc-pr-huggingface-pytorch-training"
        )
        == ""
    )

    assert (
        config.parse_dlc_developer_configs(
            "buildspec_override", "dlc-pr-huggingface-pytorch-trcomp-training"
        )
        == ""
    )
    assert (
        config.parse_dlc_developer_configs(
            "buildspec_override", "dlc-pr-huggingface-tensorflow-2-trcomp-training"
        )
        == ""
    )
    assert (
        config.parse_dlc_developer_configs("buildspec_override", "dlc-pr-pytorch-trcomp-training")
        == ""
    )

    assert (
        config.parse_dlc_developer_configs("buildspec_override", "dlc-pr-mxnet-neuron-training")
        == ""
    )
    assert (
        config.parse_dlc_developer_configs("buildspec_override", "dlc-pr-pytorch-neuron-training")
        == ""
    )
    assert (
        config.parse_dlc_developer_configs(
            "buildspec_override", "dlc-pr-tensorflow-2-neuron-training"
        )
        == ""
    )

    assert (
        config.parse_dlc_developer_configs(
            "buildspec_override", "dlc-pr-stabilityai-pytorch-training"
        )
        == ""
    )

    assert (
        config.parse_dlc_developer_configs("buildspec_override", "dlc-pr-pytorch-habana-training")
        == ""
    )
    assert (
        config.parse_dlc_developer_configs(
            "buildspec_override", "dlc-pr-tensorflow-2-habana-training"
        )
        == ""
    )

    assert config.parse_dlc_developer_configs("buildspec_override", "dlc-pr-mxnet-inference") == ""
    assert (
        config.parse_dlc_developer_configs("buildspec_override", "dlc-pr-pytorch-inference") == ""
    )
    assert (
        config.parse_dlc_developer_configs("buildspec_override", "dlc-pr-tensorflow-2-inference")
        == ""
    )
    assert (
        config.parse_dlc_developer_configs("buildspec_override", "dlc-pr-autogluon-inference") == ""
    )

    assert (
        config.parse_dlc_developer_configs("buildspec_override", "dlc-pr-mxnet-neuron-inference")
        == ""
    )
    assert (
        config.parse_dlc_developer_configs("buildspec_override", "dlc-pr-pytorch-neuron-inference")
        == ""
    )
    assert (
        config.parse_dlc_developer_configs(
            "buildspec_override", "dlc-pr-tensorflow-1-neuron-inference"
        )
        == ""
    )
    assert (
        config.parse_dlc_developer_configs(
            "buildspec_override", "dlc-pr-tensorflow-2-neuron-inference"
        )
        == ""
    )

    assert (
        config.parse_dlc_developer_configs(
            "buildspec_override", "dlc-pr-huggingface-tensorflow-inference"
        )
        == ""
    )
    assert (
        config.parse_dlc_developer_configs(
            "buildspec_override", "dlc-pr-huggingface-pytorch-inference"
        )
        == ""
    )
    assert (
        config.parse_dlc_developer_configs(
            "buildspec_override", "dlc-pr-huggingface-pytorch-neuron-inference"
        )
        == ""
    )
    assert (
        config.parse_dlc_developer_configs(
            "buildspec_override", "dlc-pr-stabilityai-pytorch-inference"
        )
        == ""
    )

    assert (
        config.parse_dlc_developer_configs("buildspec_override", "dlc-pr-mxnet-graviton-inference")
        == ""
    )
    assert (
        config.parse_dlc_developer_configs(
            "buildspec_override", "dlc-pr-pytorch-graviton-inference"
        )
        == ""
    )
    assert (
        config.parse_dlc_developer_configs(
            "buildspec_override", "dlc-pr-tensorflow-2-graviton-inference"
        )
        == ""
    )

    assert (
        config.parse_dlc_developer_configs("buildspec_override", "dlc-pr-mxnet-eia-inference") == ""
    )
    assert (
        config.parse_dlc_developer_configs("buildspec_override", "dlc-pr-pytorch-eia-inference")
        == ""
    )
    assert (
        config.parse_dlc_developer_configs(
            "buildspec_override", "dlc-pr-tensorflow-2-eia-inference"
        )
        == ""
    )
