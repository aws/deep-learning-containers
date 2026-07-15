"""Integration tests for the WhisperX ASR SageMaker endpoint (SYNC) — SageMaker SDK v3.

Deploys one real-time inference endpoint for the module and drives it through the
SageMaker `/invocations` alias (identical handler to `/v1/audio/transcriptions`).
The container is a FastAPI WhisperX server that expects `multipart/form-data` with a
required `file` part plus optional form fields, so each request builds a multipart
body locally and sends the raw bytes via `endpoint.invoke(body=..., content_type=...)`.

AMI pin (DESIGN decision 9): WhisperX is a CUDA 12.8 GPU image, so the endpoint MUST
run on INFERENCE_AMI_VERSION_CU12 (AL2, driver 550). The default test AMI is CU13
(driver 580); using it makes the container fail to start with a zero-log
CannotStartContainerError. Pin CU12 explicitly.

Validation is smoke-level and deterministic: transcription returns non-empty text and
diarization surfaces >=2 speakers. WhisperX output is not byte-stable, so no
exact-transcript assertions.
"""

import json
import logging
import os
import time
import uuid

import pytest
from sagemaker.core.resources import Endpoint, EndpointConfig, Model
from sagemaker.core.shapes import ContainerDefinition, ProductionVariant
from test_utils import random_suffix_name
from test_utils.constants import INFERENCE_AMI_VERSION_CU12, SAGEMAKER_ROLE

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# Single L4 GPU; matches WhisperX VRAM needs (large-v2 + align + diarization).
INSTANCE_TYPE = "ml.g6.xlarge"
# Startup warms the default Whisper model + diarization pipeline in the lifespan
# hook before /ping returns 200, so InService already implies a warm model.
STARTUP_HEALTH_CHECK_TIMEOUT = 900
# Deploy can take 10-15 min for a GPU endpoint.
DEPLOY_TIMEOUT = 1800

# Audio fixtures pre-staged in S3 (read with the test runner's credentials).
AUDIO_BUCKET = "dlc-cicd-models"
AUDIO_PREFIX = "test-fixtures/audio"
AUDIO_EN = "asr_en.wav"  # English ~15s
AUDIO_DIARIZE = "asr_diarize_2spk.wav"  # 2 speakers ~21s


def _cleanup(resources):
    """Best-effort delete for a list of v3 resource objects (None-safe).

    Leaked SageMaker endpoints keep billing, so cleanup must run even when a
    test fails — callers invoke this from a `finally` block.
    """
    for resource in resources:
        if resource is None:
            continue
        try:
            resource.delete()
        except Exception as e:
            LOGGER.warning(f"Cleanup {type(resource).__name__} failed: {e}")


def _build_multipart(audio_path: str, fields: dict) -> tuple[bytes, str]:
    """Build a multipart/form-data body with the audio `file` part + string fields.

    SageMaker InvokeEndpoint forwards the ContentType header (boundary included)
    straight through to the model server, so the client owns the encoding.
    """
    boundary = uuid.uuid4().hex
    filename = os.path.basename(audio_path)
    with open(audio_path, "rb") as f:
        audio = f.read()

    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="{filename}"\r\nContent-Type: audio/wav\r\n\r\n'.encode(),
        audio,
        b"\r\n",
    ]
    for key, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n'
            f"{value}\r\n".encode()
        )
    parts.append(f"--{boundary}--\r\n".encode())

    body = b"".join(parts)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def _invoke_transcription(endpoint, audio_path: str, retries: int = 3, **fields) -> dict:
    """Invoke /invocations with a multipart audio request and return the parsed JSON.

    The first invocation of a given code path may lazily load an alignment model,
    which can nudge past SageMaker's 60s real-time invoke cap; the model is cached
    server-side afterward, so a bounded retry rides out that cold start
    deterministically rather than flaking.
    """
    body, content_type = _build_multipart(audio_path, fields)
    last_error = None
    for attempt in range(retries):
        try:
            result = endpoint.invoke(body=body, content_type=content_type)
            return json.loads(result.body.read())
        except Exception as e:
            last_error = e
            LOGGER.warning(f"Invoke attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(30)
    raise last_error


@pytest.fixture(scope="session")
def audio_cache(aws_session, tmp_path_factory):
    """Return a downloader that fetches (and memoizes) an audio fixture locally."""
    cache_dir = str(tmp_path_factory.mktemp("whisperx-audio"))
    downloaded: dict[str, str] = {}

    def _get(name: str) -> str:
        if name not in downloaded:
            dest = os.path.join(cache_dir, name)
            LOGGER.info(f"Downloading s3://{AUDIO_BUCKET}/{AUDIO_PREFIX}/{name}")
            aws_session.s3.download_file(AUDIO_BUCKET, f"{AUDIO_PREFIX}/{name}", dest)
            downloaded[name] = dest
        return downloaded[name]

    return _get


@pytest.fixture(scope="module")
def model_endpoint(aws_session, image_uri):
    """Deploy one WhisperX real-time endpoint for the module; clean up after.

    Module-scoped so all sync tests share a single (expensive) GPU deploy.
    """
    endpoint_name = random_suffix_name("whisperx", 50)
    model_name = endpoint_name
    role_arn = aws_session.resolve_role_arn(SAGEMAKER_ROLE)

    LOGGER.info(f"Using image: {image_uri}")

    model = endpoint_config = endpoint = None
    try:
        LOGGER.info(f"Creating model: {model_name}")
        model = Model.create(
            model_name=model_name,
            primary_container=ContainerDefinition(image=image_uri),
            execution_role_arn=role_arn,
        )

        LOGGER.info(f"Creating endpoint config: {endpoint_name}")
        endpoint_config = EndpointConfig.create(
            endpoint_config_name=endpoint_name,
            production_variants=[
                ProductionVariant(
                    variant_name="AllTraffic",
                    model_name=model_name,
                    initial_instance_count=1,
                    instance_type=INSTANCE_TYPE,
                    # CUDA 12.8 image -> must use the CU12 (driver 550) AMI.
                    inference_ami_version=INFERENCE_AMI_VERSION_CU12,
                    container_startup_health_check_timeout_in_seconds=STARTUP_HEALTH_CHECK_TIMEOUT,
                ),
            ],
        )

        LOGGER.info(f"Deploying endpoint: {endpoint_name} (this may take 10-15 minutes)...")
        endpoint = Endpoint.create(
            endpoint_name=endpoint_name,
            endpoint_config_name=endpoint_name,
        )
        endpoint.wait_for_status("InService", timeout=DEPLOY_TIMEOUT)
        LOGGER.info("Endpoint deployment completed successfully")

        yield endpoint
    finally:
        _cleanup([endpoint, endpoint_config, model])


def test_endpoint_in_service(model_endpoint, audio_cache):
    """Endpoint reaches InService and /invocations answers a minimal request with JSON.

    Reaching InService (the fixture) is the primary assertion; the minimal invoke
    confirms the SageMaker /invocations route is wired and returns a parseable 200.
    """
    endpoint = model_endpoint
    audio = audio_cache(AUDIO_EN)

    body = _invoke_transcription(endpoint, audio, language="en")
    assert isinstance(body, dict), f"Expected JSON object from /invocations, got: {body!r}"
    LOGGER.info("Endpoint InService and /invocations returned a parseable response")


def test_invocations_transcription(model_endpoint, audio_cache):
    """/invocations transcribes English audio and returns non-empty text (default json)."""
    endpoint = model_endpoint
    audio = audio_cache(AUDIO_EN)

    body = _invoke_transcription(endpoint, audio, language="en")
    text = body.get("text", "")
    assert isinstance(text, str) and text.strip(), f"Transcription text is empty: {body!r}"
    LOGGER.info(f"Transcription text (len={len(text)}): {text[:120]!r}")


def test_invocations_diarization(model_endpoint, audio_cache):
    """/invocations with diarize=true + verbose_json surfaces >=2 distinct speakers."""
    endpoint = model_endpoint
    audio = audio_cache(AUDIO_DIARIZE)

    body = _invoke_transcription(
        endpoint,
        audio,
        language="en",
        response_format="verbose_json",
        diarize="true",
    )
    assert body.get("text", "").strip(), f"Diarization response has empty text: {body!r}"
    speakers = body.get("speakers")
    assert speakers, f"verbose_json missing non-empty 'speakers': {body!r}"
    assert len(speakers) >= 2, f"Expected >=2 speakers, got {speakers!r}"
    LOGGER.info(f"Diarization detected {len(speakers)} speakers: {speakers}")
