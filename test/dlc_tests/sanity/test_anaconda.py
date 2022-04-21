import pytest
from invoke.context import Context
from test import test_utils


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("N/A")
@pytest.mark.integration("anaconda_removal")
def test_repo_anaconda_not_present(image):
    """Test to see if all packages installed in the image do not come from repo.anaconda.com"""
    try:
        ctx = Context()
        container_name = test_utils.get_container_name("anaconda", image)
        test_utils.start_container(container_name, image, ctx)

        # First check to see if image has conda installed, if not, skip test since no packages installed from conda present
        conda_present = test_utils.run_cmd_on_container(container_name, ctx, "find . -name conda").stdout.strip()
        if not conda_present:
            pytest.skip(f"Image {image} does not have conda installed, skipping test.")

        # Commands are split in 2 because if warn=True, then even if first command fails silently, no error is raised
        test_utils.run_cmd_on_container(container_name, ctx, "conda list --explicit > repo_list.txt")

        grep_result = test_utils.run_cmd_on_container(
            container_name, ctx, "grep repo.anaconda.com repo_list.txt", warn=True
        ).stdout.strip()
        if grep_result:
            raise RuntimeError(
                f"Image {image} contains packages installed from repo.anaconda.com. \
                Please ensure that these packages are obtained through conda-forge or other alternatives: {grep_result}"
            )
    finally:
        test_utils.stop_and_remove_container(container_name, ctx)
