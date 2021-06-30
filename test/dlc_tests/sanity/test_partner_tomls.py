import os
import tempfile

import pytest

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
    partner_toml = os.path.join(root_dir, 'partner_developers.toml')
    parser_script = os.path.join(root_dir, 'parse_partner_developers.py')

    partner_dev_key = "partner_developer"
    test_parter = "TESTING123"

    ctx = Context()

    with tempfile.NamedTemporaryFile() as temp:
        with open(partner_toml, 'r') as partner_handle:
            for line in partner_handle:
                if line.startswith(partner_dev_key):
                    temp.write(f"{partner_dev_key} = {test_parter}")
            else:
                temp.write(line)
        output = ctx.run(f"python {parser_script} --partner_toml={temp}").stdout.strip()

    assert output == f"PARTNER_DEVELOPER: {test_parter}"
