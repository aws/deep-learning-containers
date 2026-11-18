"""Verify provenance and license records, the source layout, and the SageMaker layout."""

import importlib.util
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


# The framework sources sit where the upstream radixark/miles image keeps them: Miles' Helm
# charts set workingDir /root/miles and put these checkouts on PYTHONPATH, and its recipes
# default megatron_path to /root/Megatron-LM.
@pytest.mark.parametrize(
    "module, root",
    [
        ("miles", "/root/miles"),
        ("sglang", "/sgl-workspace/sglang/python"),
        ("radix_transfer_plugins", "/sgl-workspace/sglang/radix_transfer_plugins"),
        ("megatron.core", "/root/Megatron-LM"),
        ("torchtitan", "/root/torchtitan"),
    ],
)
def test_framework_imports_from_upstream_source_layout(module, root):
    spec = importlib.util.find_spec(module)
    assert spec is not None and spec.origin is not None, f"{module} is not installed"
    assert Path(spec.origin).resolve().is_relative_to(root), spec.origin


# The test workflow sets MILES_IMAGE_TARGET from the config's build target.
@pytest.mark.skipif(
    os.environ.get("MILES_IMAGE_TARGET") != "sagemaker", reason="SageMaker target only"
)
@pytest.mark.parametrize(
    "directory", ["/opt/ml/code", "/opt/ml/input/data", "/opt/ml/model", "/opt/ml/output"]
)
def test_sagemaker_path_exists(directory):
    assert os.path.isdir(directory)
