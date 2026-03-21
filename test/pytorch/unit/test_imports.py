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
def test_import(run_in_container, package):
    run_in_container(f"python -c 'import {package}'")
