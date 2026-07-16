"""Verify environment variables, binaries, and versions set in the Dockerfile."""

import os
import shutil

import pytest


class TestContainerEnv:
    def test_dlc_container_type(self):
        assert os.environ.get("DLC_CONTAINER_TYPE") == "training"

    def test_pythondontwritebytecode(self):
        assert os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"

    def test_pythonunbuffered(self):
        assert os.environ.get("PYTHONUNBUFFERED") == "1"

    def test_lang(self):
        assert os.environ.get("LANG") == "C.UTF-8"

    def test_efa_nccl_runtime_env(self):
        assert os.environ.get("FI_PROVIDER") == "efa"
        assert os.environ.get("NCCL_SOCKET_IFNAME") == "eth0"


class TestPath:
    @pytest.mark.parametrize("entry", ["/opt/venv/bin", "/opt/amazon/openmpi/bin"])
    def test_path_entry(self, entry):
        assert entry in os.environ["PATH"]

    @pytest.mark.parametrize("entry", ["/opt/amazon/efa/bin", "/usr/local/cuda/bin"])
    def test_gpu_path_entry(self, entry):
        assert entry in os.environ["PATH"]

    def test_ld_library_path_openmpi(self):
        assert "/opt/amazon/openmpi/lib" in os.environ.get("LD_LIBRARY_PATH", "")


class TestBinaries:
    # ray + wget are required by the KubeRay workflow (dashboard/job API + probes).
    @pytest.mark.parametrize("binary", ["python", "ray", "wget", "grep", "mpirun", "nvcc"])
    def test_on_path(self, binary):
        assert shutil.which(binary) is not None, f"{binary} not on PATH"


class TestVersions:
    def test_ray_version_matches_expected(self):
        expected = os.environ.get("EXPECTED_FRAMEWORK_VERSION", "")
        if not expected:
            pytest.skip("EXPECTED_FRAMEWORK_VERSION not set")
        import ray

        assert ray.__version__ == expected

    def test_torch_is_cuda_build(self):
        import torch

        assert "cu" in torch.__version__
