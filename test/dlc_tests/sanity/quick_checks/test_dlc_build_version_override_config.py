import pytest

from src import config


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("dlc_build_version_override_config")
def test_build_version_override_configuration():
    """
    Ensure that defaults are set back to normal before merge
    """
    assert config.parse_dlc_build_version_override_configs("dlc-pr-autogluon") == ""
    assert config.parse_dlc_build_version_override_configs("dlc-pr-mxnet") == ""
    assert config.parse_dlc_build_version_override_configs("dlc-pr-pytorch") == ""
    assert config.parse_dlc_build_version_override_configs("dlc-pr-tensorflow-1") == ""
    assert config.parse_dlc_build_version_override_configs("dlc-pr-tensorflow-2") == ""

    assert config.parse_dlc_build_version_override_configs("dlc-pr-huggingface-pytorch") == ""
    assert config.parse_dlc_build_version_override_configs("dlc-pr-huggingface-tensorflow") == ""

    assert config.parse_dlc_build_version_override_configs("dlc-pr-pytorch-habana") == ""
    assert config.parse_dlc_build_version_override_configs("dlc-pr-tensorflow-2-habana") == ""

    assert config.parse_dlc_build_version_override_configs("dlc-pr-mxnet-neuron") == ""
    assert config.parse_dlc_build_version_override_configs("dlc-pr-pytorch-neuron") == ""
    assert config.parse_dlc_build_version_override_configs("dlc-pr-tensorflow-1-neuron") == ""
    assert config.parse_dlc_build_version_override_configs("dlc-pr-tensorflow-2-neuron") == ""
    assert config.parse_dlc_build_version_override_configs("dlc-pr-huggingface-pytorch-neuron") == ""
    assert config.parse_dlc_build_version_override_configs("dlc-pr-huggingface-tensorflow-neuron") == ""

    assert config.parse_dlc_build_version_override_configs("dlc-pr-huggingface-pytorch-trcomp") == ""
    assert config.parse_dlc_build_version_override_configs("dlc-pr-huggingface-tensorflow-2-trcomp") == ""

    assert config.parse_dlc_build_version_override_configs("dlc-pr-mxnet-graviton") == ""
    assert config.parse_dlc_build_version_override_configs("dlc-pr-pytorch-graviton") == ""
    assert config.parse_dlc_build_version_override_configs("dlc-pr-tensorflow-2-graviton") == ""

    assert config.parse_dlc_build_version_override_configs("dlc-pr-mxnet-eia") == ""
    assert config.parse_dlc_build_version_override_configs("dlc-pr-pytorch-eia") == ""
    assert config.parse_dlc_build_version_override_configs("dlc-pr-tensorflow-1-eia") == ""
    assert config.parse_dlc_build_version_override_configs("dlc-pr-tensorflow-2-eia") == ""
