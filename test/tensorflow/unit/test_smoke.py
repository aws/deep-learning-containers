"""Smoke tests: TF imports and runs a real op; CUDA libraries are loadable.

Unit tests run on no-GPU CodeBuild runners, so we cannot assert
`list_physical_devices("GPU")` returns a non-empty list — the host has no GPU
even though the image was built for CUDA. Instead, validate that TF was
BUILT with CUDA support and that CUDA libraries are PRESENT and LOADABLE.
Real GPU compute is verified later by `sagemaker-test` on actual GPU
instances.
"""

import ctypes
import os
import shutil

import pytest

IS_CUDA = os.path.isdir("/usr/local/cuda")
cuda_only = pytest.mark.skipif(not IS_CUDA, reason="CUDA-only test")
cpu_only = pytest.mark.skipif(IS_CUDA, reason="CPU-only test")


def test_tensorflow_matmul_runs():
    """Run a tiny matmul on whatever device TF picks (CPU on no-GPU runner).
    Asserts the op actually executes and the shape is right."""
    import tensorflow as tf

    a = tf.random.normal([256, 256])
    b = tf.random.normal([256, 256])
    c = tf.linalg.matmul(a, b)
    assert tuple(c.shape) == (256, 256)


def test_cpu_devices_detected():
    """CPU device should always be detected, regardless of image flavor."""
    import tensorflow as tf

    cpus = tf.config.list_physical_devices("CPU")
    assert len(cpus) > 0, "no CPU devices detected"


@cuda_only
def test_tensorflow_built_with_cuda():
    """TF wheel must be a CUDA-capable build. This is a build-time fact;
    works even when the test runner has no physical GPU."""
    import tensorflow as tf

    assert tf.test.is_built_with_cuda(), "TF wheel was not compiled with CUDA support"


@cuda_only
def test_cudart_loadable():
    """CUDA runtime library must be present and loadable.
    Catches missing or broken libcudart linkage."""
    ctypes.CDLL("libcudart.so")  # raises OSError if missing


@cuda_only
def test_cudnn_loadable():
    """cuDNN runtime library must be present and loadable.
    Catches missing or wrong-SOVERSION libcudnn linkage."""
    ctypes.CDLL("libcudnn.so.9")  # raises OSError if missing


@cuda_only
def test_nvcc_on_path():
    """nvcc binary should be on PATH on CUDA images
    (we install cuda-nvcc-${MAJOR_MINOR} via dnf in runtime-base)."""
    assert shutil.which("nvcc") is not None, "nvcc not found on PATH"


@cpu_only
def test_no_cuda_directory_on_cpu_image():
    """The CPU image must not contain /usr/local/cuda — guards against
    accidental CUDA leakage from the pytorch-cpu index workaround."""
    assert not os.path.isdir("/usr/local/cuda"), (
        "/usr/local/cuda exists on CPU image — base image leak?"
    )
