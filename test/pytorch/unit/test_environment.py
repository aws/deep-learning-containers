"""Verify environment variables, critical binaries, and NCCL/EFA setup."""

import ctypes
import os
import shutil
import subprocess

import pytest

IS_GPU = os.path.isdir("/usr/local/cuda")
gpu_only = pytest.mark.skipif(not IS_GPU, reason="GPU-only test")


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

    @pytest.mark.parametrize("directory", ["/opt/venv/bin", "/opt/amazon/openmpi/bin"])
    def test_path_contains(self, directory):
        assert directory in os.environ["PATH"], f"{directory} not in PATH"

    @gpu_only
    @pytest.mark.parametrize("directory", ["/opt/amazon/efa/bin", "/usr/local/cuda/bin"])
    def test_path_contains_gpu(self, directory):
        assert directory in os.environ["PATH"], f"{directory} not in PATH"

    @pytest.mark.parametrize("directory", ["/opt/amazon/openmpi/lib", "/usr/local/lib"])
    def test_ld_library_path_contains(self, directory):
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        assert directory in ld, f"{directory} not in LD_LIBRARY_PATH"

    @gpu_only
    @pytest.mark.parametrize(
        "directory", ["/opt/amazon/ofi-nccl/lib64", "/opt/amazon/efa/lib", "/usr/local/cuda/lib64"]
    )
    def test_ld_library_path_contains_gpu(self, directory):
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        assert directory in ld, f"{directory} not in LD_LIBRARY_PATH"


class TestBinaries:
    """Verify critical binaries are on PATH and executable."""

    @pytest.mark.parametrize("binary", ["python", "torchrun", "deepspeed", "mpirun", "sshd"])
    def test_binary_on_path(self, binary):
        assert shutil.which(binary) is not None, f"{binary} not found on PATH"

    @gpu_only
    @pytest.mark.parametrize("binary", ["fi_info", "nvcc"])
    def test_binary_on_path_gpu(self, binary):
        assert shutil.which(binary) is not None, f"{binary} not found on PATH"


@gpu_only
class TestNCCLAndEFA:
    """Verify NCCL and OFI NCCL plugin are properly installed."""

    def test_ofi_nccl_plugin_exists(self):
        assert os.path.isfile("/opt/amazon/ofi-nccl/lib64/libnccl-net.so")

    def test_efa_libfabric_provider(self):
        out = subprocess.check_output(["fi_info", "--version"], text=True, stderr=subprocess.STDOUT)
        assert "libfabric" in out.lower()


@gpu_only
class TestCuDNN:
    """Verify cuDNN runtime libraries are present and loadable."""

    def test_cudnn_loadable(self):
        ctypes.CDLL("libcudnn.so.9")


@gpu_only
class TestCudaRuntime:
    """Verify CUDA runtime library is loadable (required by NCCL OFI plugin)."""

    def test_cudart_loadable(self):
        ctypes.CDLL("libcudart.so")
