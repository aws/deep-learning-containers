"""Verify key Python packages import successfully."""

import importlib
import os

import pytest

IS_CUDA = os.path.isdir("/usr/local/cuda")

COMMON_PACKAGES = [
    "tensorflow",
    "keras",
    "sagemaker",
    "sagemaker_tensorflow_container",
    "boto3",
    "requests",
    "yaml",
    "packaging",
]

GPU_PACKAGES = [
    "nvidia.cudnn",
    "nvidia.nccl",
]


@pytest.mark.parametrize("package", COMMON_PACKAGES)
def test_import(package):
    importlib.import_module(package)


@pytest.mark.skipif(not IS_CUDA, reason="CUDA-only package")
@pytest.mark.parametrize("package", GPU_PACKAGES)
def test_import_gpu(package):
    importlib.import_module(package)


def test_tensorflow_version_prefix():
    import tensorflow as tf

    assert tf.__version__.startswith("2.21"), f"expected tf 2.21*, got {tf.__version__}"


def test_sagemaker_sdk_v3():
    # SDK v3 dropped the `sagemaker.__version__` module attribute that v2 exposed.
    # Use importlib.metadata, which reads from the installed package metadata.
    import importlib.metadata

    version = importlib.metadata.version("sagemaker")
    assert version.startswith("3."), f"expected sagemaker 3.x, got {version}"
