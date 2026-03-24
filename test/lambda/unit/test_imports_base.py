"""Verify key packages import successfully — base image."""

import importlib

import pytest

REQUIRED_PACKAGES = [
    "awslambdaric",
    "boto3",
    "requests",
    "pip_licenses",
]


@pytest.mark.parametrize("package", REQUIRED_PACKAGES)
def test_import(package):
    importlib.import_module(package)
