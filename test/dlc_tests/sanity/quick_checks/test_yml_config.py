import os

import pytest
import yaml

from test_utils import is_pr_context


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("release_images.yml")
@pytest.mark.skipif(
    not is_pr_context(),
    reason="This test is only needed to validate release_images configs in PRs.",
)
def test_release_images_yml():
    """
    Simple test to ensure release images yml file is loadable
    """
    # Look up the path until deep-learning-containers is our base directory
    dlc_base_dir = os.getcwd()
    while os.path.basename(dlc_base_dir) != "deep-learning-containers":
        dlc_base_dir = os.path.split(dlc_base_dir)[0]

    release_images_yml_file = os.path.join(dlc_base_dir, "release_images.yml")

    with open(release_images_yml_file, "r") as release_imgs_handle:
        try:
            yaml.safe_load(release_imgs_handle)
        except yaml.YAMLError as e:
            raise RuntimeError(
                f"Failed to load {release_images_yml_file} via pyyaml package. Please check the contents of the file, correct errors and retry."
            ) from e
