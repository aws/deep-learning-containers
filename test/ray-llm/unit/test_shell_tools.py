"""Verify shell tools and CUDA headers vLLM needs at runtime."""

import os
import shutil
from pathlib import Path

import pytest

BINARIES = ["which", "gcc", "g++", "make", "nvcc", "wget", "curl"]


@pytest.mark.parametrize("binary", BINARIES)
def test_binary_on_path(binary):
    assert shutil.which(binary), f"{binary} missing from PATH"


def test_cc_env_var():
    assert os.environ.get("CC") == "/usr/bin/gcc"


def test_cxx_env_var():
    assert os.environ.get("CXX") == "/usr/bin/g++"


def test_curand_header_present():
    assert Path("/usr/local/cuda/include/curand.h").exists()
