"""Verify environment variables, critical binaries, and NCCL/EFA setup."""

import ctypes
import os
import shutil
import subprocess
import sys

import pytest


class TestContainerEnv:
    """Verify container-level environment variables set in the Dockerfile."""

    def test_dlc_container_type(self):
        assert os.environ.get("DLC_CONTAINER_TYPE") == "training"

    def test_pythondontwritebytecode(self):
        assert os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"

    def test_pythonunbuffered(self):
        assert os.environ.get("PYTHONUNBUFFERED") == "1"

    def test_lang(self):
        assert os.environ.get("LANG") == "C.UTF-8"


class TestPath:
    """Verify PATH and LD_LIBRARY_PATH contain required directories."""

    @pytest.mark.parametrize(
        "directory",
        ["/opt/venv/bin", "/opt/amazon/openmpi/bin", "/opt/amazon/efa/bin", "/usr/local/cuda/bin"],
    )
    def test_path_contains(self, directory):
        assert directory in os.environ["PATH"], f"{directory} not in PATH"

    @pytest.mark.parametrize(
        "directory",
        [
            "/opt/amazon/ofi-nccl/lib64",
            "/opt/amazon/openmpi/lib",
            "/opt/amazon/efa/lib",
            "/usr/local/cuda/lib64",
            "/usr/local/lib",
        ],
    )
    def test_ld_library_path_contains(self, directory):
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        assert directory in ld, f"{directory} not in LD_LIBRARY_PATH"


class TestBinaries:
    """Verify critical binaries are on PATH and executable."""

    @pytest.mark.parametrize(
        "binary", ["python", "torchrun", "deepspeed", "mpirun", "fi_info", "sshd", "nvcc"]
    )
    def test_binary_on_path(self, binary):
        assert shutil.which(binary) is not None, f"{binary} not found on PATH"


class TestNCCLAndEFA:
    """Verify NCCL and OFI NCCL plugin are properly installed."""

    def test_ofi_nccl_plugin_exists(self):
        assert os.path.isfile("/opt/amazon/ofi-nccl/lib64/libnccl-net.so")

    def test_efa_libfabric_provider(self):
        out = subprocess.check_output(["fi_info", "--version"], text=True, stderr=subprocess.STDOUT)
        assert "libfabric" in out.lower()


class TestCuDNN:
    """Verify cuDNN runtime libraries are present and loadable."""

    def test_cudnn_lib_exists(self):
        cudnn_dir = os.path.join(
            sys.prefix,
            "lib",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
            "site-packages",
            "nvidia",
            "cudnn",
            "lib",
        )
        libs = [f for f in os.listdir(cudnn_dir) if f.startswith("libcudnn")]
        assert len(libs) > 0, f"No cuDNN libraries found in {cudnn_dir}"

    def test_cudnn_loadable(self):
        ctypes.CDLL("libcudnn.so.9")


class TestCudaRuntime:
    """Verify CUDA runtime library is loadable (required by NCCL OFI plugin)."""

    def test_cudart_loadable(self):
        ctypes.CDLL("libcudart.so")
