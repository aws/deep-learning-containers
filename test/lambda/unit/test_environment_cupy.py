"""Verify cupy-image-specific environment — cupy image."""

import os

import pytest


@pytest.mark.parametrize("var", ["CUPY_CACHE_DIR", "NUMBA_CACHE_DIR"])
def test_jit_cache_under_tmp(var):
    """CuPy and Numba JIT-compile at runtime and default their cache to $HOME."""
    value = os.environ.get(var, "")
    assert value.startswith("/tmp/"), f"{var}={value!r}"
