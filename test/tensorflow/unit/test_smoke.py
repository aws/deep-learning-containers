"""Smoke test: TensorFlow can run a real op and detect devices."""

import os

import pytest

IS_CUDA = os.path.isdir("/usr/local/cuda")
cuda_only = pytest.mark.skipif(not IS_CUDA, reason="CUDA-only test")
cpu_only = pytest.mark.skipif(IS_CUDA, reason="CPU-only test")


def test_tensorflow_matmul_runs():
    """Run a tiny matmul on whatever device TF picks. Asserts the op
    actually executes (returns a non-zero tensor) and the shape is right."""
    import tensorflow as tf

    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    b = tf.constant([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])  # (3, 2)
    c = tf.linalg.matmul(a, b)
    assert tuple(c.shape) == (2, 2)
    # All four entries are positive — a passing matmul yields non-zero output.
    assert tf.reduce_min(c).numpy() > 0


@cuda_only
def test_gpu_devices_detected():
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    assert len(gpus) > 0, "no GPU devices detected on CUDA image"


@cpu_only
def test_no_gpu_devices_on_cpu_image():
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    assert len(gpus) == 0, f"unexpected GPU devices on CPU image: {gpus}"


def test_cpu_devices_detected():
    import tensorflow as tf

    cpus = tf.config.list_physical_devices("CPU")
    assert len(cpus) > 0, "no CPU devices detected"
