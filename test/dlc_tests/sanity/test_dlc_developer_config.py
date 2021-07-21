import os

import pytest
import toml


from test.test_utils import is_dlc_cicd_context


@pytest.mark.model("N/A")
@pytest.mark.integration("dlc_developer_config")
@pytest.mark.skipif(not is_dlc_cicd_context(), reason="Test relies on CB env variables and should be run on CB")
def test_developer_configuration():
    """
    Ensure that defaults are set back to normal before merge
    """
    root_dir = os.getenv("CODEBUILD_SRC_DIR")
    dev_config = os.path.join(root_dir, "dlc_developer_config.toml")
    dev_config_contents = toml.load(dev_config)

    # Check dev settings
    assert _get_option(dev_config_contents, "dev", "partner_developer") == ""
    assert _get_option(dev_config_contents, "dev", "ei_mode") is False
    assert _get_option(dev_config_contents, "dev", "neuron_mode") is False
    assert _get_option(dev_config_contents, "dev", "benchmark_mode") is False

    # Check build settings
    assert _get_option(dev_config_contents, "build", "skip_frameworks") == []
    assert _get_option(dev_config_contents, "build", "datetime_tag") is True
    assert _get_option(dev_config_contents, "build", "do_build") is True

    # Check test settings
    assert _get_option(dev_config_contents, "test", "efa_tests") is False
    assert _get_option(dev_config_contents, "test", "sanity_tests") is True
    assert _get_option(dev_config_contents, "test", "sagemaker_tests") is False
    assert _get_option(dev_config_contents, "test", "ecs_tests") is True
    assert _get_option(dev_config_contents, "test", "eks_tests") is True
    assert _get_option(dev_config_contents, "test", "ec2_tests") is True
    assert _get_option(dev_config_contents, "test", "use_scheduler") is False


def _get_option(toml_contents, section, option):
    return toml_contents.get(section, {}).get(option)
