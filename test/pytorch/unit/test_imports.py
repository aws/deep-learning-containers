"""Verify key Python packages import successfully."""

import pytest

REQUIRED_PACKAGES = [
    "torch",
    "torchvision",
    "torchaudio",
    "deepspeed",
    "flash_attn",
    "transformer_engine",
    "boto3",
    "requests",
    "yaml",
    "packaging",
]


@pytest.mark.parametrize("package", REQUIRED_PACKAGES)
def test_import(container_exec, package):
    container_exec(f"python -c 'import {package}'")
