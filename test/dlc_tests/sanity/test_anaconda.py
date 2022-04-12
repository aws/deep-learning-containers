import pytest
from invoke.context import Context
from test import test_utils

@pytest.mark.model("N/A")
def test_repo_anaconda_not_present(image):
    """ Test to see if all packages installed in the image do not come from repo.anaconda.com """
    ctx = Context()
    container_name = test_utils.get_container_name("anaconda", image)
    test_utils.start_container(container_name, image, ctx)

    output = test_utils.run_cmd_on_container(container_name, ctx, "conda list --explicit | grep repo.anaconda.com").stdout.strip()
    if output:
        pytest.fail(f"Image contains packages installed from repo.anaconda.com: {output}")

