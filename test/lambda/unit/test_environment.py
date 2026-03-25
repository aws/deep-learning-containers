"""Verify environment variables set in the Dockerfile."""

import os

import pytest


class TestContainerEnv:
    """Container-level environment variables."""

    def test_dlc_container_type(self):
        assert os.environ.get("DLC_CONTAINER_TYPE") == "general"

    def test_pythondontwritebytecode(self):
        assert os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"

    def test_lambda_task_root(self):
        assert os.environ.get("LAMBDA_TASK_ROOT") == "/var/task"

    def test_lambda_runtime_dir(self):
        assert os.environ.get("LAMBDA_RUNTIME_DIR") == "/var/runtime"

    def test_lang(self):
        assert os.environ.get("LANG") == "en_US.UTF-8"

    def test_tz(self):
        assert os.environ.get("TZ") == ":/etc/localtime"


class TestPath:
    """PATH and LD_LIBRARY_PATH contain required directories."""

    @pytest.mark.parametrize("directory", ["/var/lang/bin"])
    def test_path_contains(self, directory):
        assert directory in os.environ["PATH"]

    @pytest.mark.parametrize(
        "directory",
        [
            "/var/lang/lib",
            "/lib64",
            "/usr/lib64",
            "/var/runtime",
            "/var/runtime/lib",
            "/var/task",
            "/var/task/lib",
            "/opt/lib",
            "/usr/local/cuda/lib64",
            "/x86_64-bottlerocket-linux-gnu/sys-root/usr/lib/nvidia",
        ],
    )
    def test_ld_library_path_contains(self, directory):
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        assert directory in ld, f"{directory} not in LD_LIBRARY_PATH"

    def test_pythonpath_contains_runtime(self):
        assert "/var/runtime" in os.environ.get("PYTHONPATH", "")


class TestWorkdir:
    """Working directory exists."""

    def test_var_task_exists(self):
        assert os.path.isdir("/var/task")
