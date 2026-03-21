"""Verify Docker labels, environment variables, and critical binaries."""

import pytest


class TestDockerLabels:
    """Verify image metadata labels used by SageMaker and other services."""

    def test_framework_label(self, run_in_container):
        out = run_in_container(
            "python -c \"import os; print(os.environ.get('DLC_CONTAINER_TYPE', ''))\""
        )
        assert out == "training"

    def test_entrypoint_runs(self, run_in_container):
        """Verify entrypoint.sh executes and passes through to the command."""
        out = run_in_container("echo entrypoint_ok")
        assert out == "entrypoint_ok"


class TestEnvironment:
    """Verify PATH and LD_LIBRARY_PATH contain required directories."""

    @pytest.mark.parametrize(
        "directory",
        [
            "/opt/venv/bin",
            "/opt/amazon/openmpi/bin",
            "/opt/amazon/efa/bin",
            "/usr/local/cuda/bin",
        ],
    )
    def test_path_contains(self, run_in_container, directory):
        out = run_in_container("echo $PATH")
        assert directory in out, f"{directory} not in PATH: {out}"

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
    def test_ld_library_path_contains(self, run_in_container, directory):
        out = run_in_container("echo $LD_LIBRARY_PATH")
        assert directory in out, f"{directory} not in LD_LIBRARY_PATH: {out}"


class TestBinaries:
    """Verify critical binaries are on PATH and executable."""

    @pytest.mark.parametrize(
        "binary",
        [
            "python",
            "torchrun",
            "deepspeed",
            "mpirun",
            "fi_info",
            "sshd",
            "nvcc",
        ],
    )
    def test_binary_on_path(self, run_in_container, binary):
        run_in_container(f"which {binary}")


class TestNCCLAndEFA:
    """Verify NCCL and OFI NCCL plugin are properly installed."""

    def test_ofi_nccl_plugin_exists(self, run_in_container):
        """OFI NCCL plugin is critical for EFA-based multi-node training."""
        run_in_container("test -f /opt/amazon/ofi-nccl/lib/libnccl-net.so")

    def test_nccl_library_loadable(self, run_in_container):
        """Verify NCCL can be loaded by PyTorch."""
        run_in_container(
            "python -c 'import torch; assert torch.cuda.nccl.is_available((torch.randn(1),))'"
        )

    def test_efa_libfabric_provider(self, run_in_container):
        """Verify EFA libfabric provider is available."""
        out = run_in_container("fi_info --version")
        assert "libfabric" in out.lower()
