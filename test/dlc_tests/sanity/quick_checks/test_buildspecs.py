import os
import re

import pytest


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("buildspecs")
def test_train_inference_buildspec():
    """
    Walk through all buildspecs. Ensure the following:
    1. "training" or "inference is in the path
    2. if "training" is in the path, make sure "inference" is not in the file
    3. if "inference" is in the path, make sure "training" is not in the file
    """
    # Define directories where non-image-type buildspecs are allowed
    EXEMPT_DIRS = ("deep-learning-containers", "src", "eks_infrastructure")
    # Look up the path until deep-learning-containers is our base directory
    dlc_base_dir = os.getcwd()
    while os.path.basename(dlc_base_dir) != "deep-learning-containers":
        dlc_base_dir = os.path.split(dlc_base_dir)[0]

    # Regex definitions for buildspec matching
    buildspec_pattern = re.compile(r"buildspec\S*\.yml")
    training_pattern = re.compile(r"training", re.IGNORECASE)
    inference_pattern = re.compile(r"inference", re.IGNORECASE)

    # Navigate through files and look for matches
    for root, dirnames, filenames in os.walk(dlc_base_dir):
        for filename in filenames:
            if buildspec_pattern.search(filename):
                buildspec_path = os.path.join(dlc_base_dir, root, filename)

                # Only look in directories that are not exempted from the search
                if os.path.basename(buildspec_path) not in EXEMPT_DIRS:
                    _assert_single_image_type_buildspec(buildspec_path, inference_pattern, training_pattern)


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
                ), f"Found inference reference in training buildspec {buildspec_path}"
    elif "inference" in buildspec_path:
        with open(buildspec_path) as inf_buildspec_handle:
            for line in inf_buildspec_handle:
                assert not training_pattern.search(
                    line
                ), f"Found training reference in training buildspec {buildspec_path}"
    else:
        raise RuntimeError(
            f"Buildspec {buildspec_path} is not under a training dir nor an inference dir! Please correct this and retry."
        )
