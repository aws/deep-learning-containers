"""Verify environment variables, paths, and telemetry scripts for llama.cpp DLC."""

import os

import pytest


class TestLibraryPath:
    """Verify LD_LIBRARY_PATH contains required directories."""

    @pytest.mark.parametrize(
        "directory",
        ["/usr/local/lib", "/usr/local/cuda/lib64"],
    )
    def test_ld_library_path_contains(self, directory):
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        assert directory in ld, f"{directory} not in LD_LIBRARY_PATH"


class TestWorkdir:
    """Verify /workspace directory exists for model storage."""

    def test_workspace_exists(self):
        assert os.path.isdir("/workspace"), "/workspace directory not found"


class TestDLCEnvVars:
    """Verify DLC-standard environment variables are set."""

    def test_dlc_container_type(self):
        assert os.environ.get("DLC_CONTAINER_TYPE"), "DLC_CONTAINER_TYPE not set"

    def test_lang(self):
        assert os.environ.get("LANG") == "C.UTF-8"

    def test_lc_all(self):
        assert os.environ.get("LC_ALL") == "C.UTF-8"


class TestTelemetry:
    """Verify DLC telemetry scripts are present."""

    @pytest.mark.parametrize(
        "path",
        [
            "/usr/local/bin/deep_learning_container.py",
            "/usr/local/bin/bash_telemetry.sh",
        ],
    )
    def test_telemetry_script_exists(self, path):
        assert os.path.isfile(path), f"{path} not found"


class TestEntrypoint:
    """Verify the container entrypoint script."""

    ENTRYPOINT_CANDIDATES = [
        "/usr/local/bin/sagemaker_entrypoint.sh",
        "/usr/local/bin/dockerd_entrypoint.sh",
        "/usr/bin/serve",
    ]

    def _find_entrypoint(self):
        for path in self.ENTRYPOINT_CANDIDATES:
            if os.path.isfile(path):
                return path
        return None

    def test_entrypoint_exists(self):
        ep = self._find_entrypoint()
        assert ep is not None, f"No entrypoint found at any of {self.ENTRYPOINT_CANDIDATES}"

    def test_entrypoint_is_executable(self):
        ep = self._find_entrypoint()
        if ep is None:
            pytest.skip("No entrypoint found")
        assert os.access(ep, os.X_OK), f"{ep} is not executable"
