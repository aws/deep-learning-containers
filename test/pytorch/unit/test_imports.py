"""Verify key Python packages import successfully."""

import importlib

import pytest

COMMON_PACKAGES = [
    "torch",
    "torchvision",
    "torchaudio",
    "deepspeed",
    "boto3",
    "requests",
    "yaml",
    "packaging",
]


@pytest.mark.parametrize("package", COMMON_PACKAGES)
def test_import(package):
    importlib.import_module(package)
