import os
import re

import pytest

from test.test_utils import is_pr_context, get_repository_local_path


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("buildspecs")
@pytest.mark.skipif(
    not is_pr_context(),
    reason="This tests to ensure train/inference buildspecs are set up as expected in PRs.",
)
def test_train_inference_buildspec():
    """
    Walk through all buildspecs. Ensure the following:
    1. "training" or "inference" is in the path
    2. if "training" is in the path, make sure "inference" is not in the file
    3. if "inference" is in the path, make sure "training" is not in the file

    Framework buildspecs are expected to follow buildspec<any_chars>.yml
    """
    dlc_base_dir = get_repository_local_path()

    # Regex definitions for buildspec matching
    buildspec_pattern = re.compile(r"buildspec\S*\.yml")
    training_pattern = re.compile(r"training", re.IGNORECASE)
    inference_pattern = re.compile(r"inference", re.IGNORECASE)

    # Navigate through files and look for matches
    for root, dirnames, filenames in os.walk(dlc_base_dir):
        for filename in filenames:
            if buildspec_pattern.match(filename):
                buildspec_path = os.path.join(dlc_base_dir, root, filename)

                # Don't look for framework buildspecs in the top level directory - these are not framework buildspecs
                if os.path.split(buildspec_path)[0] != dlc_base_dir:
                    _assert_single_image_type_buildspec(
                        buildspec_path, inference_pattern, training_pattern
                    )


def _assert_single_image_type_buildspec(buildspec_path, inference_pattern, training_pattern):
    """
    Isolate condition for checking whether an buildspec is consistent with its image type (training or inference).
    Require that images are nested under training or inference, if not, raise error.
    """
    if "training" in buildspec_path:
        with open(buildspec_path) as trn_buildspec_handle:
            for line in trn_buildspec_handle:
                assert not inference_pattern.search(
                    line
                ), f"Found inference reference in training buildspec {buildspec_path}. Please check the file and remove them."
    elif "inference" in buildspec_path:
        with open(buildspec_path) as inf_buildspec_handle:
            for line in inf_buildspec_handle:
                assert not training_pattern.search(
                    line
                ), f"Found training reference in inference buildspec {buildspec_path}. Please check the file and remove them."
    else:
        raise RuntimeError(
            f"Buildspec {buildspec_path} is not under a training dir nor an inference dir! Please correct this and retry."
        )
