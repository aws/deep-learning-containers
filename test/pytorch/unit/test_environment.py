"""Verify environment variables, critical binaries, and NCCL/EFA setup."""

import os
import shutil
import subprocess

import pytest


class TestEnvironment:
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
            "/opt/amazon/ofi-nccl/lib",
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
        assert os.path.isfile("/opt/amazon/ofi-nccl/lib/libnccl-net.so")

    def test_nccl_library_loadable(self):
        import torch

        assert torch.cuda.nccl.is_available((torch.randn(1),))

    def test_efa_libfabric_provider(self):
        out = subprocess.check_output(["fi_info", "--version"], text=True, stderr=subprocess.STDOUT)
        assert "libfabric" in out.lower()
