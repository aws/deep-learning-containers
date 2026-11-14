"""Verify the CUDA image ships a TensorFlow build with native Blackwell (sm_120) code.

The upstream PyPI wheel stops at sm_89 + compute_90 PTX, which fails or falls back
to JIT compilation on G7/G7e. The CUDA image installs a source-built wheel instead.
"""

import os

import pytest

IS_CUDA = os.path.isdir("/usr/local/cuda")

pytestmark = pytest.mark.skipif(not IS_CUDA, reason="CUDA image only")


def test_build_targets_include_sm120():
    import tensorflow as tf

    caps = tf.sysconfig.get_build_info()["cuda_compute_capabilities"]
    assert "compute_120" in caps, f"TensorFlow built for {caps}, missing compute_120"


def test_build_targets_include_sm100():
    import tensorflow as tf

    caps = tf.sysconfig.get_build_info()["cuda_compute_capabilities"]
    assert "sm_100" in caps, f"TensorFlow built for {caps}, missing sm_100 (B200/GB200)"
