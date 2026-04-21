"""Verify filesystem layout: SageMaker paths, EFA, NCCL, GDRCopy, venv."""

import os

import pytest

IS_CUDA = os.path.isdir("/usr/local/cuda")
cuda_only = pytest.mark.skipif(not IS_CUDA, reason="CUDA-only test")

SAGEMAKER_PATHS = [
    "/opt/ml/input/data",
    "/opt/ml/model",
    "/opt/ml/output",
    "/opt/ml/code",
]


@pytest.mark.parametrize("path", SAGEMAKER_PATHS)
def test_sagemaker_path_exists(path):
    if not os.path.isdir("/opt/ml"):
        pytest.skip("SageMaker paths only exist in sagemaker image")
    assert os.path.isdir(path), f"{path} does not exist"


def test_openmpi_binary_exists():
    assert os.access("/opt/amazon/openmpi/bin/mpirun", os.X_OK)


@cuda_only
def test_efa_binary_exists():
    assert os.access("/opt/amazon/efa/bin/fi_info", os.X_OK)


@cuda_only
def test_nccl_config():
    with open("/etc/nccl.conf") as f:
        content = f.read()
    assert "NCCL_DEBUG=INFO" in content
    assert "NCCL_SOCKET_IFNAME" in content


@cuda_only
def test_gdrcopy_lib():
    assert os.path.isfile("/usr/local/lib/libgdrapi.so")


def test_venv_exists():
    assert os.path.isdir("/opt/venv/bin")
