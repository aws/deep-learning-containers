"""Verify installed package versions match pins from versions.env."""

import os
import re
import subprocess

import pytest

# Detect GPU vs CPU image by checking for CUDA, then pick the right versions file.
_WORKDIR = os.environ.get("DLC_WORKDIR", "/workdir")
IS_CUDA = os.path.isdir("/usr/local/cuda")
_VERSIONS_FILE = "versions-cuda.env" if IS_CUDA else "versions-cpu.env"
VERSIONS_ENV = os.path.join(_WORKDIR, "docker", "pytorch", _VERSIONS_FILE)
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

COMMON_PACKAGE_VERSION_MAP = {
    "torch": "TORCH_VERSION",
}

GPU_PACKAGE_VERSION_MAP = {
    "flash_attn": "FLASH_ATTN_VERSION",
    "transformer_engine": "TRANSFORMER_ENGINE_VERSION",
}


@pytest.mark.parametrize(
    "package,env_var",
    list(COMMON_PACKAGE_VERSION_MAP.items()),
    ids=list(COMMON_PACKAGE_VERSION_MAP.keys()),
)
def test_version(package, env_var):
    mod = __import__(package)
    actual = mod.__version__
    expected = ENV[env_var]
    assert actual.startswith(expected), f"{package}: expected {expected}*, got {actual}"


@cuda_only
@pytest.mark.parametrize(
    "package,env_var",
    list(GPU_PACKAGE_VERSION_MAP.items()),
    ids=list(GPU_PACKAGE_VERSION_MAP.keys()),
)
def test_version_gpu(package, env_var):
    mod = __import__(package)
    actual = mod.__version__
    expected = ENV[env_var]
    assert actual.startswith(expected), f"{package}: expected {expected}*, got {actual}"


def test_deepspeed_version():
    import deepspeed

    expected = ENV["DEEPSPEED_VERSION"]
    assert deepspeed.__version__.startswith(expected), (
        f"deepspeed: expected {expected}*, got {deepspeed.__version__}"
    )


def test_python_version():
    expected = ENV["PYTHON_VERSION"]
    out = subprocess.check_output(["python", "--version"], text=True).strip()
    assert expected in out, f"Expected Python {expected}, got {out}"


@cuda_only
def test_cuda_version():
    import torch

    expected = ".".join(ENV["CUDA_VERSION"].split(".")[:2])
    assert torch.version.cuda.startswith(expected), (
        f"Expected CUDA {expected}*, got {torch.version.cuda}"
    )
