"""Helpers for the ray-llm EC2 single-GPU test.

The workflow uses .github/actions/download-model to fetch + extract the model
tarball into a cache dir, then hands us the path via the MODEL_DIR env var.
We `docker run --gpus all` the container with the model + repo-side config.yaml
bind-mounted, wait for Ray Serve health, and hand the fixture to the tests.
"""

import logging
import os
import subprocess
import time

import pytest
import requests

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

SERVE_PORT = 8000
HEALTH_TIMEOUT = 900
HEALTH_INTERVAL = 5
REQUEST_TIMEOUT = 120

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_YAML = os.path.join(SCRIPT_DIR, "config.yaml")


def start_container(image_uri, model_dir):
    """Start Ray Serve container with the model + repo-side config.yaml mounted.

    Model weights live at /opt/ml/weights, config at /opt/ml/config/config.yaml —
    non-overlapping mount points to avoid file-inside-ro-dir portability issues.
    """
    cmd = [
        "docker",
        "run",
        "-d",
        "--gpus",
        "all",
        "--shm-size=8g",
        "-p",
        f"{SERVE_PORT}:{SERVE_PORT}",
        "-v",
        f"{model_dir}:/opt/ml/weights:ro",
        "-v",
        f"{CONFIG_YAML}:/opt/ml/config/config.yaml:ro",
        image_uri,
        "serve",
        "run",
        "/opt/ml/config/config.yaml",
    ]
    LOGGER.info(f"Starting container: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    container_id = result.stdout.strip()
    LOGGER.info(f"Container started: {container_id[:12]}")
    return container_id


def wait_for_health(port=SERVE_PORT, timeout=HEALTH_TIMEOUT, interval=HEALTH_INTERVAL):
    endpoint = f"http://localhost:{port}/-/healthz"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(endpoint, timeout=5)
            if resp.status_code == 200:
                LOGGER.info("Ray Serve is healthy")
                return
        except requests.ConnectionError:
            pass
        time.sleep(interval)
    raise TimeoutError(f"Ray Serve did not become healthy within {timeout}s")


def get_container_logs(container_id):
    result = subprocess.run(
        ["docker", "logs", "--tail", "200", container_id],
        capture_output=True,
        text=True,
    )
    return result.stdout + result.stderr


def stop_container(container_id):
    LOGGER.info(f"Stopping container {container_id[:12]}")
    subprocess.run(["docker", "rm", "-f", container_id], capture_output=True)


@pytest.fixture(scope="module")
def container(image_uri):
    model_dir = os.environ.get("MODEL_DIR")
    if not model_dir or not os.path.isdir(model_dir):
        pytest.fail(f"MODEL_DIR env var missing or not a directory: {model_dir!r}")

    container_id = start_container(image_uri, model_dir)
    try:
        wait_for_health()
    except TimeoutError:
        LOGGER.error(f"Container logs:\n{get_container_logs(container_id)}")
        stop_container(container_id)
        pytest.fail("Ray Serve health check timed out")

    yield {"container_id": container_id, "port": SERVE_PORT}

    stop_container(container_id)
