"""Verify provenance and license records, the source layout, and the SageMaker layout."""

import importlib.metadata
import importlib.util
import os
import re
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


# The LGPL-3.0 Python packages the image ships. Each needs its source listed at the
# installed version.
@pytest.mark.parametrize("name", ["easydict", "psycopg", "psycopg-binary"])
def test_lgpl_package_source_listed(name):
    version = importlib.metadata.version(name)
    lines = Path("/root/THIRD_PARTY_SOURCE_CODE_URLS").read_text().splitlines()
    assert any(line.split()[:2] == [name, version] for line in lines), f"{name} {version}"


# Every binary the Dockerfile builds or downloads is tracked, with a source URL.
@pytest.mark.parametrize(
    "name", ["FFmpeg", "GDRCopy", "nvtop", "nccl-tests", "kubectl", "Helm", "sgl-model-gateway"]
)
def test_binary_tracked(name):
    assert f"\n** {name};" in Path("/root/BUILD_FROM_SOURCE_PACKAGES_LICENCES").read_text()
    lines = Path("/root/THIRD_PARTY_SOURCE_CODE_URLS").read_text().splitlines()
    assert any(line.startswith(f"{name} ") for line in lines), name


# No source file carries a notice for a license that isn't open source. The Dockerfile
# leaves out the upstream files that do: torchtitan's FLUX model and SGLang's
# Hunyuan3D mesh extensions.
RESTRICTED_LICENSE = re.compile(
    rb"Llama ?[0-9.]* Community License|Gemma Terms of Use|Non-?Commercial License",
    re.IGNORECASE,
)


@pytest.mark.parametrize(
    "root", ["/root/miles", "/root/Megatron-LM", "/root/torchtitan", "/sgl-workspace/sglang"]
)
def test_source_has_no_restricted_license_notice(root):
    hits = []
    for directory, subdirectories, files in os.walk(root):
        subdirectories[:] = [d for d in subdirectories if d not in (".git", "__pycache__")]
        for name in files:
            path = os.path.join(directory, name)
            if os.path.islink(path):
                continue
            with open(path, "rb") as f:
                data = f.read()
            # Skip binary files, as grep -I does.
            if b"\0" not in data[:8192] and RESTRICTED_LICENSE.search(data):
                hits.append(path)
    assert not hits, hits


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
