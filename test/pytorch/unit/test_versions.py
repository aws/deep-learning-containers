"""Verify installed package versions match expected pins."""

import pytest

EXPECTED_VERSIONS = {
    "torch": "2.10.0",
    "torchvision": "0.25.0",
    "torchaudio": "2.10.0",
    "flash_attn": "2.7.4.post1",
    "transformer_engine": "2.3.0",
}


@pytest.mark.parametrize("package,expected", EXPECTED_VERSIONS.items())
def test_version(run_in_container, package, expected):
    actual = run_in_container(f"python -c 'import {package}; print({package}.__version__)'")
    assert actual.startswith(expected), f"{package}: expected {expected}*, got {actual}"


def test_deepspeed_version(run_in_container):
    # deepspeed prints warnings to stdout before the version
    out = run_in_container("python -c 'import deepspeed; print(deepspeed.__version__)'")
    version = out.strip().splitlines()[-1]
    assert version.startswith("0.16.7"), f"deepspeed: expected 0.16.7*, got {version}"


def test_python_version(run_in_container):
    out = run_in_container("python --version")
    assert "3.12" in out
