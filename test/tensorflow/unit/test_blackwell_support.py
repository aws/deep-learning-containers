"""Verify the CUDA image ships a TensorFlow build with native Blackwell code.

The upstream PyPI wheel stops at sm_89 + compute_90 PTX, which fails or falls back
to JIT compilation on G7/G7e (sm_120). The CUDA image installs a source-built wheel
instead. These tests inspect the library that ships, not only the build config.
"""

import glob
import os
import shutil
import subprocess

import pytest

IS_CUDA = os.path.isdir("/usr/local/cuda")

pytestmark = pytest.mark.skipif(not IS_CUDA, reason="CUDA image only")


def _cuobjdump():
    candidates = [shutil.which("cuobjdump"), "/usr/local/cuda/bin/cuobjdump"]
    try:
        import triton

        # triton bundles its own cuobjdump.
        candidates += glob.glob(
            os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "cuobjdump")
        )
    except ImportError:
        pass
    for path in candidates:
        if path and os.access(path, os.X_OK):
            return path
    pytest.fail("cuobjdump not found in the image")


def test_build_targets_include_blackwell():
    import tensorflow as tf

    caps = tf.sysconfig.get_build_info()["cuda_compute_capabilities"]
    assert "sm_100" in caps, f"TensorFlow built for {caps}, missing sm_100 (B200/GB200/B300)"
    # compute_120 already gives the nvcc-built kernels sm_120 code plus PTX; the
    # explicit sm_120 gives native G7/G7e code to the MLIR kernels generated at
    # build time (kernel_gen emits only PTX for compute_XY entries).
    assert "sm_120" in caps, f"TensorFlow built for {caps}, missing sm_120 (G7/G7e)"
    assert "compute_120" in caps, f"TensorFlow built for {caps}, missing compute_120 (G7/G7e)"


@pytest.mark.parametrize("arch", ["sm_100", "sm_120"])
def test_tensorflow_library_has_blackwell_cubins(arch):
    import tensorflow as tf

    lib = os.path.join(os.path.dirname(tf.__file__), "libtensorflow_cc.so.2")
    assert os.path.isfile(lib), f"{lib} not found"
    elfs = subprocess.run(
        [_cuobjdump(), "--list-elf", lib], check=True, capture_output=True, text=True
    ).stdout
    assert arch in elfs, f"{lib} has no {arch} cubins"
