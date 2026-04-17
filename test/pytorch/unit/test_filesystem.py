"""Verify filesystem layout: SageMaker paths, EFA, NCCL, GDRCopy, venv."""

import os

import pytest

SAGEMAKER_PATHS = [
    "/opt/ml/input/data",
    "/opt/ml/model",
    "/opt/ml/output",
    "/opt/ml/code",
]

EFA_BINARIES = [
    "/opt/amazon/efa/bin/fi_info",
    "/opt/amazon/openmpi/bin/mpirun",
]


@pytest.mark.parametrize("path", SAGEMAKER_PATHS)
def test_sagemaker_path_exists(path):
    if not os.path.isdir("/opt/ml"):
        pytest.skip("SageMaker paths only exist in sagemaker image")
    assert os.path.isdir(path), f"{path} does not exist"


@pytest.mark.parametrize("binary", EFA_BINARIES)
def test_efa_binary_exists(binary):
    assert os.access(binary, os.X_OK), f"{binary} not found or not executable"


def test_nccl_config():
    with open("/etc/nccl.conf") as f:
        content = f.read()
    assert "NCCL_DEBUG=INFO" in content
    assert "NCCL_SOCKET_IFNAME" in content


def test_gdrcopy_lib():
    assert os.path.isfile("/usr/local/lib/libgdrapi.so")


def test_venv_exists():
    assert os.path.isdir("/opt/venv/bin")
