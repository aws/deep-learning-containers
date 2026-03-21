"""Verify installed package versions match pins from versions.env."""

import os
import re

import pytest

# Parse versions.env as the single source of truth
VERSIONS_ENV = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "docker", "pytorch", "versions.env"
)


def _parse_versions_env():
    """Read shell export lines from versions.env into a dict."""
    versions = {}
    with open(VERSIONS_ENV) as f:
        for line in f:
            m = re.match(r'^export\s+(\w+)="?([^"$]+)"?', line.strip())
            if m:
                versions[m.group(1)] = m.group(2)
    return versions


ENV = _parse_versions_env()

# Map package import names to versions.env variable names
PACKAGE_VERSION_MAP = {
    "torch": "TORCH_VERSION",
    "torchvision": "TORCHVISION_VERSION",
    "torchaudio": "TORCHAUDIO_VERSION",
    "flash_attn": "FLASH_ATTN_VERSION",
    "transformer_engine": "TRANSFORMER_ENGINE_VERSION",
}


@pytest.mark.parametrize(
    "package,env_var",
    [(pkg, var) for pkg, var in PACKAGE_VERSION_MAP.items()],
    ids=list(PACKAGE_VERSION_MAP.keys()),
)
def test_version(run_in_container, package, env_var):
    expected = ENV[env_var]
    actual = run_in_container(f"python -c 'import {package}; print({package}.__version__)'")
    assert actual.startswith(expected), f"{package}: expected {expected}*, got {actual}"


def test_deepspeed_version(run_in_container):
    expected = ENV["DEEPSPEED_VERSION"]
    out = run_in_container("python -c 'import deepspeed; print(deepspeed.__version__)'")
    # deepspeed may print warnings before the version
    version = out.strip().splitlines()[-1]
    assert version.startswith(expected), f"deepspeed: expected {expected}*, got {version}"


def test_python_version(run_in_container):
    expected = ENV["PYTHON_VERSION"]
    out = run_in_container("python --version")
    assert expected in out, f"Expected Python {expected}, got {out}"


def test_cuda_version(run_in_container):
    expected_major_minor = ".".join(ENV["CUDA_VERSION"].split(".")[:2])
    out = run_in_container("python -c 'import torch; print(torch.version.cuda)'")
    assert out.startswith(expected_major_minor), f"Expected CUDA {expected_major_minor}*, got {out}"
