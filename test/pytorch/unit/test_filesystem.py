"""Verify filesystem layout: SageMaker paths, EFA, NCCL, GDRCopy, venv."""

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
def test_sagemaker_path_exists(container_exec, path):
    container_exec(f"test -d {path}")


@pytest.mark.parametrize("binary", EFA_BINARIES)
def test_efa_binary_exists(container_exec, binary):
    container_exec(f"test -x {binary}")


def test_nccl_config(container_exec):
    out = container_exec("cat /etc/nccl.conf")
    assert "NCCL_DEBUG=INFO" in out


def test_gdrcopy_lib(container_exec):
    container_exec("test -f /usr/local/lib/libgdrapi.so")


def test_venv_exists(container_exec):
    container_exec("test -d /opt/venv/bin")
