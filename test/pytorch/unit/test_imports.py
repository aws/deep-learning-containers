"""Verify key Python packages import successfully."""

import importlib
import os

import pytest

IS_GPU = os.path.isdir("/usr/local/cuda")

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

GPU_PACKAGES = [
    "flash_attn",
    "transformer_engine",
]


@pytest.mark.parametrize("package", COMMON_PACKAGES)
def test_import(package):
    importlib.import_module(package)


@pytest.mark.skipif(not IS_GPU, reason="GPU-only package")
@pytest.mark.parametrize("package", GPU_PACKAGES)
def test_import_gpu(package):
    importlib.import_module(package)
