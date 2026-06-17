"""Verify installed package versions match pins from versions.env."""

import os
import re
import subprocess

import pytest

IS_CUDA = os.path.isdir("/usr/local/cuda")
cuda_only = pytest.mark.skipif(not IS_CUDA, reason="CUDA-only test")


def _parse_versions_env(path):
    versions = {}
    with open(path) as f:
        for line in f:
            m = re.match(r'^export\s+(\w+)="?([^"$]+)"?', line.strip())
            if m:
                versions[m.group(1)] = m.group(2)
    return versions


@pytest.fixture(scope="session")
def versions_env(request):
    pt_version = request.config.getoption("--pytorch-version")
    workdir = request.config.getoption("--workdir")
    versions_file = "versions-cuda.env" if IS_CUDA else "versions-cpu.env"
    path = os.path.join(workdir, "docker", "pytorch", pt_version, versions_file)
    return _parse_versions_env(path)


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
def test_version(package, env_var, versions_env):
    mod = __import__(package)
    actual = mod.__version__
    expected = versions_env[env_var]
    assert actual.startswith(expected), f"{package}: expected {expected}*, got {actual}"


@cuda_only
@pytest.mark.parametrize(
    "package,env_var",
    list(GPU_PACKAGE_VERSION_MAP.items()),
    ids=list(GPU_PACKAGE_VERSION_MAP.keys()),
)
def test_version_gpu(package, env_var, versions_env):
    mod = __import__(package)
    actual = mod.__version__
    expected = versions_env[env_var]
    assert actual.startswith(expected), f"{package}: expected {expected}*, got {actual}"


def test_deepspeed_version(versions_env):
    import deepspeed

    expected = versions_env["DEEPSPEED_VERSION"]
    assert deepspeed.__version__.startswith(expected), (
        f"deepspeed: expected {expected}*, got {deepspeed.__version__}"
    )


def test_python_version(versions_env):
    expected = versions_env["PYTHON_VERSION"]
    out = subprocess.check_output(["python", "--version"], text=True).strip()
    assert expected in out, f"Expected Python {expected}, got {out}"


@cuda_only
def test_cuda_version(versions_env):
    import torch

    expected = ".".join(versions_env["CUDA_VERSION"].split(".")[:2])
    assert torch.version.cuda.startswith(expected), (
        f"Expected CUDA {expected}*, got {torch.version.cuda}"
    )
