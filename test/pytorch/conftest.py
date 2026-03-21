"""PyTorch DLC test fixtures."""

import subprocess

import pytest


def pytest_addoption(parser):
    parser.addoption("--image-uri", required=True, help="Docker image URI to test")


@pytest.fixture(scope="session")
def image_uri(request):
    return request.config.getoption("--image-uri")


@pytest.fixture(scope="session")
def run_in_container(image_uri):
    """Run a command inside the container and return stdout."""

    def _run(cmd, user=None, gpu=False):
        docker_cmd = ["docker", "run", "--rm"]
        if gpu:
            docker_cmd.append("--gpus=all")
        if user:
            docker_cmd.extend(["--user", user])
        docker_cmd.extend([image_uri, "bash", "-c", cmd])
        result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {cmd}\nstderr: {result.stderr}")
        return result.stdout.strip()

    return _run
