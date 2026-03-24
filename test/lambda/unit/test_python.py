"""Verify Python runtime, symlinks, and version."""

import subprocess
import sys


def test_python_version():
    assert sys.version_info >= (3, 13), f"Expected Python >= 3.13, got {sys.version_info}"


def test_python3_symlink():
    out = subprocess.check_output(["python3", "--version"], text=True).strip()
    assert f"{sys.version_info.major}.{sys.version_info.minor}" in out


def test_python_symlink():
    out = subprocess.check_output(["python", "--version"], text=True).strip()
    assert f"{sys.version_info.major}.{sys.version_info.minor}" in out
