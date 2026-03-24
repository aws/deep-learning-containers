"""Verify CUDA runtime libraries are present and loadable."""

import ctypes
import os


def test_cuda_lib_dir_exists():
    assert os.path.isdir("/usr/local/cuda/lib64")


def test_cudart_loadable():
    ctypes.CDLL("libcudart.so.12")
