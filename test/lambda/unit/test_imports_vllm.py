"""Verify key packages import successfully — vllm image (CPU-safe imports)."""

import importlib

import pytest

REQUIRED_PACKAGES = [
    "awslambdaric",
    "boto3",
    "numpy",
    "safetensors",
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
]


@pytest.mark.parametrize("package", REQUIRED_PACKAGES)
def test_import(package):
    importlib.import_module(package)
