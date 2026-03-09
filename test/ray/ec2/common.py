# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Shared constants, helpers, fixtures, and test implementations for Ray EC2 container tests.

CPU and GPU test modules import from here, setting only DEVICE and DOCKER_RUN_FLAGS.

Unlike SageMaker tests which deploy to real endpoints, EC2 tests run the container
locally via `docker run`, hit the Ray Serve HTTP endpoint on port 8000, and validate
responses using the same validators from ray.utils.
"""

import logging
import os
import subprocess
import tarfile
import tempfile
import time

import pytest
import requests
from ray.utils import (
    IRIS_SAMPLES,
    MIN_MNIST_ACCURACY,
    S3_BUCKET,
    S3_PREFIX,
    SENTIMENT_SAMPLES,
    download_all_test_images,
    make_all_digit_pngs,
    make_all_sine_wavs,
    validate_audio_response,
    validate_densenet_response,
    validate_iris_response,
    validate_mnist_response,
    validate_sentiment_response,
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

SERVE_PORT = 8000
HEALTH_ENDPOINT = f"http://localhost:{SERVE_PORT}/-/healthz"
HEALTH_TIMEOUT = 180  # seconds to wait for Ray Serve to become healthy
HEALTH_INTERVAL = 5  # seconds between health checks


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


def build_models(device):
    """Build the MODELS dict parameterized by device ("cpu" or "gpu").

    Each model defines:
      - s3_key: path under S3_BUCKET/S3_PREFIX to the model.tar.gz
      - docker_args: extra args for `docker run` (e.g. CLI arg for serve target)
      - env: environment variables to pass to the container (mirrors SageMaker Model env)
      - description: human-readable label for logs
    """
    num_gpus = "0" if device == "cpu" else "1"
    return {
        "cv-densenet": {
            "s3_key": f"cv-densenet/{device}/model.tar.gz",
            "docker_args": [],  # auto-detect /opt/ml/model/config.yaml
            "env": {},
            "description": "DenseNet image classification (mounted config, auto-detect)",
        },
        "mnist-direct-app": {
            "s3_key": f"mnist-direct-app/{device}/model.tar.gz",
            "docker_args": ["deployment:app"],  # CLI arg: module:app import
            "env": {"RAYSERVE_NUM_GPUS": num_gpus},
            "description": "MNIST via direct app import (CLI arg deployment:app)",
        },
        "tabular": {
            "s3_key": f"tabular/{device}/model.tar.gz",
            "docker_args": ["/opt/ml/model/config.yaml"],  # CLI arg: explicit config path
            "env": {},
            "description": "Iris tabular (CLI arg explicit config path + runtime requirements.txt)",
        },
        "nlp": {
            "s3_key": f"nlp/{device}/model.tar.gz",
            "docker_args": [],  # auto-detect /opt/ml/model/config.yaml
            "env": {},
            "description": "DistilBERT sentiment (mounted config, auto-detect)",
        },
        "audio-ffmpeg": {
            "s3_key": f"audio-ffmpeg/{device}/model.tar.gz",
            "docker_args": [],  # auto-detect /opt/ml/model/config.yaml
            "env": {},
            "description": "Wav2Vec2 audio transcription (mounted config, ffmpeg backend)",
        },
    }


# ---------------------------------------------------------------------------
# Container lifecycle helpers
# ---------------------------------------------------------------------------


def download_and_extract_model(aws_session, model_name, device):
    """Download model tarball from S3 and extract to a temp directory.

    Returns the path to the extracted model directory (to be mounted at /opt/ml/model).
    """
    models = build_models(device)
    s3_key = f"{S3_PREFIX}/{models[model_name]['s3_key']}"
    model_dir = tempfile.mkdtemp(prefix=f"ray-ec2-{model_name}-")
    tarball_path = os.path.join(model_dir, "model.tar.gz")

    LOGGER.info(f"Downloading s3://{S3_BUCKET}/{s3_key} -> {tarball_path}")
    aws_session.s3.download_file(S3_BUCKET, s3_key, tarball_path)

    LOGGER.info(f"Extracting to {model_dir}")
    with tarfile.open(tarball_path, "r:gz") as tar:
        tar.extractall(model_dir)
    os.remove(tarball_path)

    return model_dir


def start_container(image_uri, model_dir, model_name, device, docker_run_flags=None):
    """Start a Docker container running Ray Serve with the given model.

    Args:
        image_uri: Full ECR image URI.
        model_dir: Local path to extracted model (mounted at /opt/ml/model).
        model_name: Model key from build_models().
        device: "cpu" or "gpu".
        docker_run_flags: Extra flags for docker run (e.g. ["--gpus", "all"]).

    Returns:
        container_id: Docker container ID.
    """
    models = build_models(device)
    docker_args = models[model_name]["docker_args"]
    env = models[model_name].get("env", {})

    cmd = [
        "docker",
        "run",
        "-d",
        "-p",
        f"{SERVE_PORT}:{SERVE_PORT}",
        "-v",
        f"{model_dir}:/opt/ml/model",
    ]
    for key, val in env.items():
        cmd.extend(["-e", f"{key}={val}"])
    if docker_run_flags:
        cmd.extend(docker_run_flags)
    cmd.append(image_uri)
    cmd.extend(docker_args)

    LOGGER.info(f"Starting container: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    container_id = result.stdout.strip()
    LOGGER.info(f"Container started: {container_id[:12]}")
    return container_id


def wait_for_health(timeout=HEALTH_TIMEOUT, interval=HEALTH_INTERVAL):
    """Poll the Ray Serve health endpoint until it returns 200 or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(HEALTH_ENDPOINT, timeout=5)
            if resp.status_code == 200:
                LOGGER.info("Ray Serve is healthy")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(interval)
    raise TimeoutError(f"Ray Serve did not become healthy within {timeout}s")


def get_container_logs(container_id):
    """Get docker logs for debugging."""
    result = subprocess.run(
        ["docker", "logs", container_id],
        capture_output=True,
        text=True,
    )
    return result.stdout + result.stderr


def stop_container(container_id):
    """Stop and remove a Docker container."""
    LOGGER.info(f"Stopping container {container_id[:12]}")
    subprocess.run(["docker", "rm", "-f", container_id], capture_output=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_model_name_fixture():
    """Create the model_name fixture (parametrized per test)."""

    @pytest.fixture(scope="function")
    def model_name(request):
        return request.param

    return model_name


def make_container_fixture(device, docker_run_flags=None):
    """Create the container fixture: download model, start container, health check, cleanup.

    Yields a dict with container_id and model_dir for the test to use.
    """

    @pytest.fixture(scope="function")
    def container(aws_session, image_uri, model_name):
        model_dir = download_and_extract_model(aws_session, model_name, device)
        container_id = start_container(
            image_uri,
            model_dir,
            model_name,
            device,
            docker_run_flags,
        )
        try:
            wait_for_health()
        except TimeoutError:
            logs = get_container_logs(container_id)
            LOGGER.error(f"Container logs:\n{logs}")
            stop_container(container_id)
            pytest.fail(f"Ray Serve health check timed out for {model_name}")

        yield {"container_id": container_id, "model_dir": model_dir}

        stop_container(container_id)

    return container


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def post_json(payload):
    """Send a JSON payload to the Ray Serve endpoint and return parsed response."""
    resp = requests.post(
        f"http://localhost:{SERVE_PORT}/",
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def post_bytes(data, content_type):
    """Send raw bytes to the Ray Serve endpoint and return parsed response."""
    resp = requests.post(
        f"http://localhost:{SERVE_PORT}/",
        data=data,
        headers={"Content-Type": content_type},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Test implementations
# ---------------------------------------------------------------------------


def run_test_cv_densenet(container):
    """DenseNet image classification — kitten and flower images, validate top-5 structure."""
    images = download_all_test_images()

    for img_name, img_data in images.items():
        result = post_bytes(img_data, "image/jpeg")
        LOGGER.info(f"cv-densenet {img_name} response: {result}")

        err = validate_densenet_response(result)
        assert not err, f"cv-densenet {img_name}: {err}"

        top = result["predictions"][0]
        LOGGER.info(
            f"  {img_name} -> {top['class_name']} "
            f"(class_id={top['class_id']}, prob={top['probability']:.4f})"
        )


def run_test_mnist_direct_app(container):
    """MNIST via deployment:app — classify all 10 digits, enforce accuracy threshold."""
    digit_pngs = make_all_digit_pngs()
    correct = 0
    total = len(digit_pngs)

    for digit_id, png_data in sorted(digit_pngs.items()):
        result = post_bytes(png_data, "image/png")
        LOGGER.info(f"mnist-direct-app digit_{digit_id} response: {result}")

        err = validate_mnist_response(result)
        assert not err, f"mnist-direct-app digit_{digit_id}: {err}"

        pred = result["prediction"]
        conf = result["confidence"]
        if pred == digit_id:
            correct += 1
            LOGGER.info(f"  digit_{digit_id} -> predicted={pred} confidence={conf:.4f}")
        else:
            LOGGER.warning(
                f"  digit_{digit_id} -> predicted={pred} expected={digit_id} confidence={conf:.4f}"
            )

    accuracy = int(correct * 100 / total)
    LOGGER.info(f"mnist-direct-app accuracy: {correct}/{total} ({accuracy}%)")
    assert accuracy >= MIN_MNIST_ACCURACY, (
        f"MNIST accuracy {accuracy}% below threshold {MIN_MNIST_ACCURACY}%"
    )


def run_test_tabular(container):
    """Iris classification — 6 samples, validate predicted species.

    Also verifies that the container installed runtime requirements.txt
    (the tabular model tarball includes code/requirements.txt).
    """
    # Verify runtime requirements.txt was installed by checking container logs
    logs = get_container_logs(container["container_id"])
    assert "Installing packages from /opt/ml/model/code/requirements.txt" in logs, (
        "Container did not install runtime requirements.txt from model tarball"
    )

    for features, expected, desc in IRIS_SAMPLES:
        payload = {"features": features}
        result = post_json(payload)
        LOGGER.info(f"tabular {desc} response: {result}")

        err = validate_iris_response(result)
        assert not err, f"tabular {desc}: {err}"

        pred = result["prediction"]
        conf = result["confidence"]
        assert pred == expected, f"tabular {desc}: predicted '{pred}', expected '{expected}'"
        LOGGER.info(f"  {desc} -> {pred} ({conf:.4f})")


def run_test_nlp(container):
    """DistilBERT sentiment — 6 samples (3 pos, 3 neg), validate predicted label."""
    for text, expected_label in SENTIMENT_SAMPLES:
        payload = {"text": text}
        result = post_json(payload)
        LOGGER.info(f"nlp response for '{text}': {result}")

        err = validate_sentiment_response(result)
        assert not err, f"nlp '{text}': {err}"

        label = result["predictions"][0]["label"]
        score = result["predictions"][0]["score"]
        assert label == expected_label, (
            f"nlp '{text}': predicted '{label}', expected '{expected_label}'"
        )
        LOGGER.info(f"  '{text}' -> {label} ({score:.4f})")


def run_test_audio_ffmpeg(container):
    """Wav2Vec2 transcription — 3 sine waves, validate structure + ffmpeg backend."""
    wavs = make_all_sine_wavs()

    for name, wav_data in wavs:
        result = post_bytes(wav_data, "audio/wav")
        LOGGER.info(f"audio-ffmpeg {name} response: {result}")

        err = validate_audio_response(result, check_ffmpeg_backend=True)
        assert not err, f"audio-ffmpeg {name}: {err}"

        LOGGER.info(
            f'  {name} -> "{result["transcription"]}" (backend: {result.get("audio_backend", "?")})'
        )
