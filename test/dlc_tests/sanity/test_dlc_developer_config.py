import os
import tempfile

import pytest
import toml

from invoke.context import Context

from test.test_utils import is_dlc_cicd_context


@pytest.mark.model("N/A")
@pytest.mark.integration("partner_developer_experience")
@pytest.mark.skipif(not is_dlc_cicd_context(), reason="Test relies on CB env variables and should be run on CB")
def test_parse_partner_developers():
    """
    Simple integration test to ensure the integrity of the partner developer experience
    """
    root_dir = os.getenv("CODEBUILD_SRC_DIR")
    partner_toml = os.path.join(root_dir, "dlc_developer_config.toml")
    parser_script = os.path.join(root_dir, "parse_partner_developers.py")

    partner_dev_key = "partner_developer"
    test_parter = "TESTING123"

    ctx = Context()

    with tempfile.NamedTemporaryFile() as temp:
        with open(partner_toml, "r") as partner_handle:
            for line in partner_handle:
                if line.startswith(partner_dev_key):
                    temp.write(f"{partner_dev_key} = {test_parter}")
            else:
                temp.write(line)
        output = ctx.run(f"python {parser_script} --partner_toml={temp}").stdout.strip()

    assert output == f"PARTNER_DEVELOPER: {test_parter}"


@pytest.mark.model("N/A")
@pytest.mark.integration("dlc_developer_config")
@pytest.mark.skipif(not is_dlc_cicd_context(), reason="Test relies on CB env variables and should be run on CB")
def test_developer_configuration():
    """
    Ensure that defaults are set back to normal before merge
    """
    root_dir = os.getenv("CODEBUILD_SRC_DIR")
    partner_toml = os.path.join(root_dir, "dlc_developer_config.toml")
    toml_contents = toml.load(partner_toml)
    partner_dev_key = "partner_developer"

    # Check dev settings
    assert _get_option(toml_contents, "dev", "partner_developer") is False
    assert _get_option(toml_contents, "dev", "ei_mode") is False
    assert _get_option(toml_contents, "dev", "neuron_mode") is False
    assert _get_option(toml_contents, "dev", "benchmark_mode") is False

    # Check build settings
    assert _get_option(toml_contents, "build", "skip_frameworks") == []
    assert _get_option(toml_contents, "build", "datetime_tag") is True
    assert _get_option(toml_contents, "build", "do_build") is True

    # Check test settings
    assert _get_option(toml_contents, "test", "efa_tests") is True
    assert _get_option(toml_contents, "test", "sanity_tests") is True
    assert _get_option(toml_contents, "test", "sagemaker_tests") is True
    assert _get_option(toml_contents, "test", "ecs_tests") is True
    assert _get_option(toml_contents, "test", "eks_tests") is True
    assert _get_option(toml_contents, "test", "ec2_tests") is True
    assert _get_option(toml_contents, "test", "use_scheduler") is False


def _get_option(toml_contents, section, option):
    return toml_contents.get(section, {}).get(option)
