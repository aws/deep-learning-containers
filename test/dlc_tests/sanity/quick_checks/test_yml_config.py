import os
import re

import pytest
import yaml

from test.test_utils import is_pr_context, get_repository_local_path


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("release_images_training.yml")
@pytest.mark.skipif(
    not is_pr_context(),
    reason="This test is only needed to validate release_images configs in PRs.",
)
def test_release_images_training_yml():
    _release_images_yml_verifier(image_type="training", excluded_image_type="inference")


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("release_images_inference.yml")
@pytest.mark.skipif(
    not is_pr_context(),
    reason="This test is only needed to validate release_images configs in PRs.",
)
def test_release_images_inference_yml():
    _release_images_yml_verifier(image_type="inference", excluded_image_type="training")


def _release_images_yml_verifier(image_type, excluded_image_type):
    """
    Simple test to ensure release images yml file is loadable
    Also test that excluded_image_type is not present in the release yml file
    """
    dlc_base_dir = get_repository_local_path()

    release_images_yml_file = os.path.join(dlc_base_dir, f"release_images_{image_type}.yml")

    # Define exclude regex
    exclude_pattern = re.compile(rf"{excluded_image_type}", re.IGNORECASE)

    with open(release_images_yml_file, "r") as release_imgs_handle:
        for line in release_imgs_handle:
            assert not exclude_pattern.search(
                line
            ), f"{exclude_pattern.pattern} found in {release_images_yml_file}. Please ensure there are not conflicting job types here."
        try:
            yaml.safe_load(release_imgs_handle)
        except yaml.YAMLError as e:
            raise RuntimeError(
                f"Failed to load {release_images_yml_file} via pyyaml package. Please check the contents of the file, correct errors and retry."
            ) from e
