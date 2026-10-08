"""Verify CUDA runtime libraries are present and loadable."""

import ctypes
import glob
import os


def test_cuda_lib_dir_exists():
    assert os.path.isdir("/usr/local/cuda/lib64")


def test_cudart_loadable():
    # Load whichever libcudart major the image ships (CUDA-version-agnostic).
    candidates = glob.glob("/usr/local/cuda/lib64/libcudart.so.*") + glob.glob(
        "/usr/lib64/libcudart.so.*"
    )
    assert candidates, "no libcudart.so.* found in the image"
    ctypes.CDLL(candidates[0])
