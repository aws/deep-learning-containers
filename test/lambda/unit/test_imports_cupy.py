"""Verify key packages import successfully — cupy image."""

import importlib

import pytest

REQUIRED_PACKAGES = [
    "awslambdaric",
    "boto3",
    "cupy",
    "cvxpy",
    "numba",
    "numpy",
    "pandas",
    "scipy",
]


@pytest.mark.parametrize("package", REQUIRED_PACKAGES)
def test_import(package):
    importlib.import_module(package)
