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
"""Shared constants, helpers, fixtures, and HTTP utilities for WhisperX EC2 tests.

The GPU test module imports from here, setting only DEVICE and DOCKER_RUN_FLAGS.

Like the Ray EC2 suite, these tests run the container locally via `docker run`,
hit the WhisperX FastAPI server on port 8000, and validate the OpenAI-compatible
transcription contract. Unlike Ray, no model tarball is mounted: the Whisper
default model and wav2vec2 aligners download lazily from HuggingFace at runtime
and the pyannote diarization models are baked into the image, so the container
is started with just the image URI and the entrypoint runs uvicorn itself.
"""

import logging
import os
import subprocess
import time
from contextlib import contextmanager

import pytest
import requests

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# The WhisperX EC2 target EXPOSEs 8000 and its entrypoint runs
# `uvicorn server:app --host 0.0.0.0 --port 8000`.
WHISPERX_PORT = 8000

# First boot warm-loads the default Whisper model (downloaded from HF) and the
# baked pyannote diarization pipeline inside the FastAPI lifespan hook *before*
# uvicorn binds the socket, so /ping is only reachable once models are resident.
# Give it a generous budget.
HEALTH_TIMEOUT = 300  # seconds to wait for /ping to return 200
HEALTH_INTERVAL = 5  # seconds between health checks
# A transcription that needs word timestamps or diarization triggers a lazy
# download of the wav2vec2 aligner for the detected language on first use, so
# the request timeout must tolerate a cold HF fetch.
REQUEST_TIMEOUT = 600  # seconds to wait for a transcription response

# Audio fixtures staged in S3 (see test-fixtures/audio/ under the models bucket).
S3_BUCKET = "dlc-cicd-models"
S3_AUDIO_PREFIX = "test-fixtures/audio"
AUDIO_EN = "asr_en.wav"  # English ~15s clip
AUDIO_ZH = "asr_zh.wav"  # Chinese clip (non-English + language param)
AUDIO_DIARIZE_2SPK = "asr_diarize_2spk.wav"  # 2 distinct speakers ~21s


# ---------------------------------------------------------------------------
# Container lifecycle helpers
# ---------------------------------------------------------------------------


def start_container(image_uri, device, docker_run_flags=None, env=None):
    """Start a Docker container running the WhisperX FastAPI server.

    No model mount and no serve command: the image bakes/downloads its models
    and the ENTRYPOINT runs uvicorn on port 8000.

    Args:
        image_uri: Full ECR image URI.
        device: "gpu" or "cpu" (informational; GPU access comes from flags).
        docker_run_flags: Extra flags for docker run (e.g. ["--gpus", "all"]).
        env: Optional dict of environment variables, emitted as `-e K=V` flags
            (e.g. {"WHISPERX_DEFAULT_MODEL": "tiny"}). Lets a test pin the served
            model, alias, or override behavior at launch.

    Returns:
        (container_id, port)
    """
    cmd = [
        "docker",
        "run",
        "-d",
        "--shm-size=2g",
        "-p",
        f"{WHISPERX_PORT}:{WHISPERX_PORT}",
    ]
    for key, value in (env or {}).items():
        cmd.extend(["-e", f"{key}={value}"])
    if docker_run_flags:
        cmd.extend(docker_run_flags)
    cmd.append(image_uri)

    LOGGER.info(f"Starting {device} container: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    container_id = result.stdout.strip()
    LOGGER.info(f"Container started: {container_id[:12]}")
    return container_id, WHISPERX_PORT


def wait_for_health(port=WHISPERX_PORT, timeout=HEALTH_TIMEOUT, interval=HEALTH_INTERVAL):
    """Poll GET /ping until it returns 200, or raise TimeoutError.

    The socket is refused until the lifespan warm-load finishes, so connection
    errors are expected and simply retried.
    """
    endpoint = f"http://localhost:{port}/ping"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(endpoint, timeout=5)
            if resp.status_code == 200:
                LOGGER.info("WhisperX server is healthy")
                return True
        except requests.RequestException:
            pass
        time.sleep(interval)
    raise TimeoutError(f"WhisperX server did not become healthy within {timeout}s")


def get_container_logs(container_id):
    """Return combined stdout+stderr from `docker logs` for debugging."""
    result = subprocess.run(
        ["docker", "logs", container_id],
        capture_output=True,
        text=True,
    )
    return result.stdout + result.stderr


def stop_container(container_id):
    """Dump logs (best effort) then force-remove a Docker container."""
    LOGGER.info(f"Stopping container {container_id[:12]}")
    subprocess.run(["docker", "rm", "-f", container_id], capture_output=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_container_fixture(device, docker_run_flags=None):
    """Create the container fixture: start container, health-check, cleanup.

    Yields a dict with container_id and port for the test to use. On a health
    timeout the container logs are dumped and the container is removed before
    failing, so failures point directly at an image/server problem.
    """

    @pytest.fixture(scope="function")
    def container(image_uri):
        container_id, port = start_container(image_uri, device, docker_run_flags)
        try:
            wait_for_health(port=port)
        except TimeoutError:
            logs = get_container_logs(container_id)
            LOGGER.error(f"Container logs:\n{logs}")
            stop_container(container_id)
            pytest.fail("WhisperX server health check timed out")

        yield {"container_id": container_id, "port": port}

        stop_container(container_id)

    return container


@contextmanager
def run_container_with_env(image_uri, env, device="gpu", flags=("--gpus", "all")):
    """Start a WhisperX container with custom env vars; yield {container_id, port}.

    A context-manager sibling of make_container_fixture for tests that need a
    per-test container launched with specific env (e.g. WHISPERX_DEFAULT_MODEL=tiny
    to pin a fast-warming served model). It mirrors the fixture's start ->
    health-check -> (log dump on timeout) -> yield -> teardown flow, but wraps the
    yield in try/finally so `docker rm -f` always runs — a health timeout or a
    failing test body must never leak the container (a leak blocks the runner).

    Usage:
        with run_container_with_env(image_uri, {"WHISPERX_DEFAULT_MODEL": "tiny"}) as c:
            ... requests to c["port"] ...
    """
    container_id, port = start_container(image_uri, device, docker_run_flags=list(flags), env=env)
    try:
        try:
            wait_for_health(port=port)
        except TimeoutError:
            logs = get_container_logs(container_id)
            LOGGER.error(f"Container logs:\n{logs}")
            pytest.fail("WhisperX server health check timed out")

        yield {"container_id": container_id, "port": port}
    finally:
        stop_container(container_id)


# ---------------------------------------------------------------------------
# S3 fixture download
# ---------------------------------------------------------------------------


def download_fixture(aws_session, key, dest):
    """Download an audio fixture from S3 to a local path.

    Args:
        aws_session: The shared AWSSessionManager (provides a boto3 s3 client).
        key: Bare fixture filename under S3_AUDIO_PREFIX (e.g. "asr_en.wav").
        dest: Local path to write the downloaded file to.

    Returns:
        dest (for convenient inline use).
    """
    s3_key = f"{S3_AUDIO_PREFIX}/{key}"
    LOGGER.info(f"Downloading s3://{S3_BUCKET}/{s3_key} -> {dest}")
    aws_session.s3.download_file(S3_BUCKET, s3_key, dest)
    return dest


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def post_transcription(port, audio_path, timeout=REQUEST_TIMEOUT, **form_fields):
    """POST an audio file to /v1/audio/transcriptions as multipart/form-data.

    The audio is sent as the required `file` part. Remaining transcription
    parameters are passed as keyword args and encoded as form fields:

      - None values are dropped (the server applies its own defaults).
      - bool values become "true"/"false" (what Pydantic bool parsing expects).
      - list/tuple values are emitted as a repeated form field, which is exactly
        how the server reads the `timestamp_granularities[]` list, e.g.
        `post_transcription(port, path, response_format="verbose_json",
                            **{"timestamp_granularities[]": ["word"]})`.

    Returns the raw requests.Response so callers can assert on status code,
    headers/content-type, and body (json or text).
    """
    with open(audio_path, "rb") as fh:
        audio_bytes = fh.read()
    files = {"file": (os.path.basename(audio_path), audio_bytes, "audio/wav")}

    # A list of (key, value) tuples lets us repeat a key for list-valued fields
    # while `files` is present (requests encodes both into one multipart body).
    data = []
    for key, value in form_fields.items():
        if value is None:
            continue
        if isinstance(value, bool):
            data.append((key, "true" if value else "false"))
        elif isinstance(value, (list, tuple)):
            for item in value:
                data.append((key, str(item)))
        else:
            data.append((key, str(value)))

    return requests.post(
        f"http://localhost:{port}/v1/audio/transcriptions",
        files=files,
        data=data,
        timeout=timeout,
    )
