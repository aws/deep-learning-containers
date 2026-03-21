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
def test_sagemaker_path_exists(run_in_container, path):
    run_in_container(f"test -d {path}")


@pytest.mark.parametrize("binary", EFA_BINARIES)
def test_efa_binary_exists(run_in_container, binary):
    run_in_container(f"test -x {binary}")


def test_nccl_config(run_in_container):
    out = run_in_container("cat /etc/nccl.conf")
    assert "NCCL_DEBUG=INFO" in out


def test_gdrcopy_lib(run_in_container):
    run_in_container("test -f /usr/local/lib/libgdrapi.so")


def test_venv_exists(run_in_container):
    run_in_container("test -d /opt/venv/bin")
