"""Verify GPU-only Python packages import successfully.

TE 2.17 dlopens libcuda.so.1 at import time; libcuda is only mounted into
containers via nvidia-container-cli when a GPU is present. That means these
imports must run on the single-GPU fleet (--gpus all), not on the CPU-only
default-runner used for unit tests. The IS_CUDA skip is kept as a
belt-and-suspenders guard for CPU-variant images.
"""

import importlib
import os

import pytest

IS_CUDA = os.path.isdir("/usr/local/cuda")

GPU_PACKAGES = [
    "flash_attn",
    "transformer_engine",
]


@pytest.mark.skipif(not IS_CUDA, reason="CUDA-only package")
@pytest.mark.parametrize("package", GPU_PACKAGES)
def test_import_gpu(package):
    importlib.import_module(package)
