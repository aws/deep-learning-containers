"""Verify installed package versions match pins from versions.env."""

import pytest

# versions.env is volume-mounted at /workdir/docker/pytorch/versions.env
VERSIONS_ENV_CONTAINER_PATH = "/workdir/docker/pytorch/versions.env"


def _read_versions(container_exec):
    """Read versions.env from inside the container (volume-mounted by workflow)."""
    import re

    raw = container_exec(f"cat {VERSIONS_ENV_CONTAINER_PATH}")
    versions = {}
    for line in raw.splitlines():
        m = re.match(r'^export\s+(\w+)="?([^"$]+)"?', line.strip())
        if m:
            versions[m.group(1)] = m.group(2)
    return versions


@pytest.fixture(scope="module")
def versions(container_exec):
    return _read_versions(container_exec)


PACKAGE_VERSION_MAP = {
    "torch": "TORCH_VERSION",
    "torchvision": "TORCHVISION_VERSION",
    "torchaudio": "TORCHAUDIO_VERSION",
    "flash_attn": "FLASH_ATTN_VERSION",
    "transformer_engine": "TRANSFORMER_ENGINE_VERSION",
}


@pytest.mark.parametrize(
    "package,env_var",
    list(PACKAGE_VERSION_MAP.items()),
    ids=list(PACKAGE_VERSION_MAP.keys()),
)
def test_version(container_exec, versions, package, env_var):
    expected = versions[env_var]
    actual = container_exec(f"python -c 'import {package}; print({package}.__version__)'")
    assert actual.startswith(expected), f"{package}: expected {expected}*, got {actual}"


def test_deepspeed_version(container_exec, versions):
    expected = versions["DEEPSPEED_VERSION"]
    out = container_exec("python -c 'import deepspeed; print(deepspeed.__version__)'")
    version = out.strip().splitlines()[-1]
    assert version.startswith(expected), f"deepspeed: expected {expected}*, got {version}"


def test_python_version(container_exec, versions):
    expected = versions["PYTHON_VERSION"]
    out = container_exec("python --version")
    assert expected in out, f"Expected Python {expected}, got {out}"


def test_cuda_version(container_exec, versions):
    expected_major_minor = ".".join(versions["CUDA_VERSION"].split(".")[:2])
    out = container_exec("python -c 'import torch; print(torch.version.cuda)'")
    assert out.startswith(expected_major_minor), f"Expected CUDA {expected_major_minor}*, got {out}"
