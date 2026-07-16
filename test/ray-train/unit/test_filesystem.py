"""Verify filesystem layout: venv, EFA/NCCL/GDRCopy, nccl-tests, entrypoint, OSS licenses."""

import glob
import os

import pytest


def test_venv_bin_exists():
    assert os.path.isdir("/opt/venv/bin")


def test_entrypoint_executable():
    assert os.access("/usr/local/bin/entrypoint.sh", os.X_OK)


def test_efa_binary_exists():
    assert os.access("/opt/amazon/efa/bin/fi_info", os.X_OK)


def test_nccl_config():
    with open("/etc/nccl.conf") as f:
        assert "NCCL_DEBUG=INFO" in f.read()


def test_ofi_nccl_plugin_exists():
    assert os.path.isfile("/opt/amazon/ofi-nccl/lib64/libnccl-net.so")


def test_all_reduce_perf_exists():
    assert os.access("/usr/local/bin/all_reduce_perf", os.X_OK)


def test_cudart_in_cuda_lib64():
    assert glob.glob("/usr/local/cuda/lib64/libcudart.so*")


OSS_LICENSE_FILES = [
    "/root/PYTHON_PACKAGES_LICENSES",
    "/root/THIRD_PARTY_SOURCE_CODE_URLS",
]


@pytest.mark.parametrize("path", OSS_LICENSE_FILES)
def test_oss_license_file_exists(path):
    assert os.path.isfile(path)
