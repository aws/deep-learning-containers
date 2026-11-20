"""Verify Ray's runtime-env agent, which sets up the environment of every Miles job."""

import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import ray

AGENT_LIBS = Path(ray.__file__).parent / "_private/runtime_env/agent/thirdparty_files"


def test_runtime_env_agent_imports_the_installed_aiohttp():
    """The agent puts its own libraries ahead of site-packages. The image empties that set,
    so the agent runs the aiohttp the image installs, with its security fixes."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.path.insert(0, {str(AGENT_LIBS)!r}); "
            "import aiohttp; print(aiohttp.__version__)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == version("aiohttp")


def test_runtime_env_vars_reach_a_task():
    """The agent applies a job's runtime_env, which Miles uses to configure its workers."""
    ray.init(num_cpus=1, include_dashboard=False)
    try:

        @ray.remote(runtime_env={"env_vars": {"MILES_RUNTIME_ENV_CHECK": "applied"}})
        def read_env():
            return os.environ.get("MILES_RUNTIME_ENV_CHECK")

        assert ray.get(read_env.remote(), timeout=300) == "applied"
    finally:
        ray.shutdown()
