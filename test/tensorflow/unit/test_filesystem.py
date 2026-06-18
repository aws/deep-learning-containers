"""Verify filesystem layout: SageMaker paths, EFA, NCCL, OSS licenses, venv."""

import glob
import os
import subprocess

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


def test_openmpi_double_wrap():
    """`mpirun` is a wrapper that exec's `mpirun.real --allow-run-as-root`. Verify the
    wrapper-vs-real split is in place — single grep -c match means it's NOT
    double-wrapped (which would happen if EFA's bundled OMPI wasn't wiped before
    the from-source build)."""
    assert os.access("/opt/amazon/openmpi/bin/mpirun.real", os.X_OK)
    out = subprocess.check_output(
        ["grep", "-c", "mpirun.real", "/opt/amazon/openmpi/bin/mpirun"], text=True
    ).strip()
    assert out == "1", f"expected 1 mpirun.real reference in wrapper, got {out}"


@cuda_only
def test_efa_binary_exists():
    assert os.access("/opt/amazon/efa/bin/fi_info", os.X_OK)


@cuda_only
def test_nccl_config():
    with open("/etc/nccl.conf") as f:
        content = f.read()
    assert "NCCL_DEBUG=INFO" in content


def test_venv_exists():
    assert os.path.isdir("/opt/venv/bin")


def test_venv_has_tensorflow():
    """tensorflow / tensorflow_cpu is installed inside /opt/venv (not system site-packages)."""
    matches = glob.glob("/opt/venv/lib/python*/site-packages/tensorflow*")
    assert matches, "no tensorflow* directory under /opt/venv site-packages"


@pytest.mark.parametrize(
    "license_file",
    [
        "/root/PYTHON_PACKAGES_LICENSES",
        "/root/LINUX_PACKAGES_LICENSES",
        "/root/BUILD_FROM_SOURCE_PACKAGES_LICENCES",
        "/root/THIRD_PARTY_SOURCE_CODE_URLS",
    ],
)
def test_oss_license_file_exists(license_file):
    assert os.path.isfile(license_file), f"{license_file} does not exist"


def test_entrypoint_executable():
    assert os.access("/usr/local/bin/entrypoint.sh", os.X_OK)
