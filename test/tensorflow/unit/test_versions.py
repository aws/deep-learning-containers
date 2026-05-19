"""Verify installed package versions match pins from versions.env."""

import importlib.metadata
import json
import os
import re
import sys

import pytest

# Detect GPU vs CPU image by checking for CUDA, then pick the right versions file.
_WORKDIR = os.environ.get("DLC_WORKDIR", "/workdir")
IS_CUDA = os.path.isdir("/usr/local/cuda")
_VERSIONS_FILE = "versions-cuda.env" if IS_CUDA else "versions-cpu.env"
VERSIONS_ENV = os.path.join(_WORKDIR, "docker", "tensorflow", _VERSIONS_FILE)
cuda_only = pytest.mark.skipif(not IS_CUDA, reason="CUDA-only test")


def _parse_versions_env():
    versions = {}
    with open(VERSIONS_ENV) as f:
        for line in f:
            m = re.match(r'^export\s+(\w+)="?([^"$]+)"?', line.strip())
            if m:
                versions[m.group(1)] = m.group(2)
    return versions


ENV = _parse_versions_env()


def test_tensorflow_version():
    """The CPU image installs the `tensorflow_cpu` distribution, the CUDA image
    installs `tensorflow`. Both expose the version through importlib.metadata
    (and at runtime as `tensorflow.__version__`)."""
    dist_name = "tensorflow" if IS_CUDA else "tensorflow_cpu"
    actual = importlib.metadata.version(dist_name)
    expected = ENV["TF_VERSION"]
    # Compare just X.Y.Z (TF_VERSION is "2.21.0" — no suffix expected).
    assert actual.startswith(expected), f"{dist_name}: expected {expected}*, got {actual}"


def test_python_version():
    expected = ENV["PYTHON_VERSION"]
    actual = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert actual == expected, f"Expected Python {expected}, got {actual}"


@cuda_only
def test_cuda_version():
    """Read CUDA version from /usr/local/cuda/version.json (shipped by the
    nvidia/cuda base image). Compare just major.minor — the patch level can
    drift across base image refreshes without affecting binary compatibility."""
    with open("/usr/local/cuda/version.json") as f:
        data = json.load(f)
    actual_full = data["cuda"]["version"]
    actual = ".".join(actual_full.split(".")[:2])
    expected = ".".join(ENV["CUDA_VERSION"].split(".")[:2])
    assert actual == expected, f"Expected CUDA {expected}, got {actual_full}"
