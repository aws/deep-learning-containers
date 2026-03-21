"""Verify key Python packages import successfully."""

import importlib

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
def test_import(package):
    importlib.import_module(package)
