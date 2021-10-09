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
    assert config.parse_dlc_developer_configs("dev", "graviton_mode") is False
    assert config.parse_dlc_developer_configs("dev", "benchmark_mode") is False

    # Check build settings
    assert config.parse_dlc_developer_configs("build", "skip_frameworks") == ["mxnet", "tensorflow", "huggingface_pytorch", "huggingface_tensorflow"]
    assert config.parse_dlc_developer_configs("build", "datetime_tag") is True
    assert config.parse_dlc_developer_configs("build", "do_build") is True

    # Check test settings
    assert config.parse_dlc_developer_configs("test", "sanity_tests") is True
    assert config.parse_dlc_developer_configs("test", "sagemaker_remote_tests") == "off"
    assert config.parse_dlc_developer_configs("test", "sagemaker_local_tests") is False
    assert config.parse_dlc_developer_configs("test", "ecs_tests") is True
    assert config.parse_dlc_developer_configs("test", "eks_tests") is True
    assert config.parse_dlc_developer_configs("test", "ec2_tests") is True
    assert config.parse_dlc_developer_configs("test", "use_scheduler") is False
    assert config.parse_dlc_developer_configs("test", "safety_check_test") is False


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("dlc_developer_config")
def test_developer_config_wrappers_defaults():
    """
    Test defaults of config file wrappers
    """
    # Check test settings
    assert config.are_efa_tests_enabled() is False
    assert config.is_sanity_test_enabled() is True
    assert config.is_sm_local_test_enabled() is False
    assert config.is_sm_remote_test_enabled() is False
    assert config.is_ecs_test_enabled() is True
    assert config.is_eks_test_enabled() is True
    assert config.is_ec2_test_enabled() is True
    assert config.is_scheduler_enabled() is False
    assert config.is_safety_check_test_enabled() is True
