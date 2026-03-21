"""PyTorch DLC test fixtures.

Tests expect the workflow to start container(s) and set environment variables:

  CONTAINER_ID   — container ID for single-container tests (unit / single_gpu / multi_gpu)
  NODE0_CONTAINER_ID, NODE1_CONTAINER_ID — for multi-node tests
"""

import os
import subprocess

import pytest


def pytest_addoption(parser):
    parser.addoption("--image-uri", required=True, help="Docker image URI under test")


@pytest.fixture(scope="session")
def image_uri(request):
    return request.config.getoption("--image-uri")


@pytest.fixture(scope="session")
def container_id():
    """Return the long-lived container ID started by the workflow."""
    cid = os.environ.get("CONTAINER_ID")
    if not cid:
        pytest.skip("CONTAINER_ID not set — run via workflow")
    return cid


@pytest.fixture(scope="session")
def container_exec(container_id):
    """Execute a command inside the workflow-managed container."""

    def _exec(cmd, user=None, timeout=120):
        docker_cmd = ["docker", "exec"]
        if user:
            docker_cmd.extend(["--user", user])
        docker_cmd.extend([container_id, "bash", "-c", cmd])
        result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed (rc={result.returncode}): {cmd}\nstderr: {result.stderr}"
            )
        return result.stdout.strip()

    return _exec
