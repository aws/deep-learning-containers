"""Verify installed package versions match pins from versions.env."""

import re
import subprocess

import pytest

# versions.env is volume-mounted at /workdir/docker/pytorch/versions.env
VERSIONS_ENV = "/workdir/docker/pytorch/versions.env"


def _parse_versions_env():
    versions = {}
    with open(VERSIONS_ENV) as f:
        for line in f:
            m = re.match(r'^export\s+(\w+)="?([^"$]+)"?', line.strip())
            if m:
                versions[m.group(1)] = m.group(2)
    return versions


ENV = _parse_versions_env()

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
def test_version(package, env_var):
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


def test_cuda_version():
    import torch

    expected = ".".join(ENV["CUDA_VERSION"].split(".")[:2])
    assert torch.version.cuda.startswith(expected), (
        f"Expected CUDA {expected}*, got {torch.version.cuda}"
    )
