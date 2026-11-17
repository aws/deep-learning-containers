"""Verify provenance and license records, and the SageMaker layout."""

import os
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "name",
    [
        "PYTHON_PACKAGES_LOCK",
        "SOURCE_MANIFEST",
        "WHEEL_ARTIFACTS_OFFICIAL",
        "WHEEL_ARTIFACT_DEVIATIONS.json",
        "python/uv.lock",
    ],
)
def test_provenance_record_present(name):
    assert (Path("/opt/aws/dlc") / name).is_file()


@pytest.mark.parametrize(
    "name",
    [
        "BUILD_FROM_SOURCE_PACKAGES_LICENCES",
        "LINUX_PACKAGES_LICENSES",
        "PYTHON_PACKAGES_LICENSES",
        "THIRD_PARTY_SOURCE_CODE_URLS",
    ],
)
def test_license_record_present(name):
    assert (Path("/root") / name).is_file()


# The test workflow sets MILES_IMAGE_TARGET from the config's build target.
@pytest.mark.skipif(
    os.environ.get("MILES_IMAGE_TARGET") != "sagemaker", reason="SageMaker target only"
)
@pytest.mark.parametrize(
    "directory", ["/opt/ml/code", "/opt/ml/input/data", "/opt/ml/model", "/opt/ml/output"]
)
def test_sagemaker_path_exists(directory):
    assert os.path.isdir(directory)
