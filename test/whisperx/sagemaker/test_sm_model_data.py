"""WhisperX bring-your-own-model integration test — proves ModelDataUrl (S3) serves.

Deploys a real-time endpoint whose ``ContainerDefinition.model_data_url`` points at a
customer model tarball in S3 (a NON-default ``faster-whisper-large-v3``; the image
default is ``large-v2``). SageMaker downloads and extracts that tarball into
``/opt/ml/model`` before the container starts; the entrypoint auto-detects the
populated dir and serves it (``WHISPERX_DEFAULT_MODEL`` is left UNSET, so the
auto-detect path — not an env override — is what's exercised).

This is the SageMaker counterpart to the bring-your-own-model contract: a customer
supplies a model via ``ModelDataUrl`` and it is served, with no env var required.

The test is offline-enforced so it cannot false-pass. The ContainerDefinition sets
``HF_HUB_OFFLINE=1``, so the Hub is unreachable: reaching ``InService`` (startup
warms the model before ``/ping`` returns 200) and returning a transcription are only
possible if the ModelDataUrl tarball staged to ``/opt/ml/model`` loaded. Without the
offline pin, a silent staging failure would fall through to the ``large-v2`` default
and download it from the Hub — passing regardless, since the response carries no
model id to distinguish them. Offline is safe for startup: VAD ships in the whisperx
wheel and diarization loads from the baked local path, so only the Whisper load is
gated to the staged model.

Scope note: SageMaker exposes only ``/invocations`` (no GET /v1/models), so we assert
a real transcription rather than a served-model id. Combined with the offline pin,
that transcription is attributable to the staged S3 model. The EC2 counterpart
(``test_ec2_byo_model.py``) additionally carries a no-mount negative control; here
SageMaker's own endpoint-creation failure on a bad ModelDataUrl plays that role.

Validation is deterministic: assert 200 + non-empty text, never an exact transcript
(WhisperX output is not byte-stable). AMI pin: CUDA 12.8 image ->
INFERENCE_AMI_VERSION_CU12 (AL2, driver 550); the default CU13 AMI causes a zero-log
CannotStartContainerError.
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

INSTANCE_TYPE = "ml.g6.xlarge"
STARTUP_HEALTH_CHECK_TIMEOUT = 900
DEPLOY_TIMEOUT = 1800

# Audio fixture pre-staged in S3 (read with the test runner's credentials).
AUDIO_BUCKET = "dlc-cicd-models"
AUDIO_PREFIX = "test-fixtures/audio"
AUDIO_EN = "asr_en.wav"  # English ~15s

# Customer model tarball under test: a NON-default faster-whisper build (large-v3),
# staged flat (ct2 files at the archive root) so faster-whisper's os.path.isdir load
# path serves it from /opt/ml/model. Distinct from the image default (large-v2) so a
# successful serve is not attributable to the baked default.
MODEL_DATA_URL = "s3://dlc-cicd-models/whisperx-models/faster-whisper-large-v3.tar.gz"


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
    """Build a multipart/form-data body with the audio `file` part + string fields."""
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

    A bounded retry rides out SageMaker's 60s real-time invoke cap on a cold first
    request deterministically rather than flaking.
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


@pytest.fixture(scope="module")
def audio_en(aws_session, tmp_path_factory):
    """Download the English audio fixture once for the module."""
    dest = os.path.join(str(tmp_path_factory.mktemp("whisperx-audio")), AUDIO_EN)
    LOGGER.info(f"Downloading s3://{AUDIO_BUCKET}/{AUDIO_PREFIX}/{AUDIO_EN}")
    aws_session.s3.download_file(AUDIO_BUCKET, f"{AUDIO_PREFIX}/{AUDIO_EN}", dest)
    return dest


@pytest.fixture(scope="module")
def model_data_endpoint(aws_session, image_uri):
    """Deploy a WhisperX endpoint whose model comes from ModelDataUrl (S3); clean up after.

    No WHISPERX_DEFAULT_MODEL is set: SageMaker stages the ModelDataUrl tarball into
    /opt/ml/model and the entrypoint auto-detects it. Module-scoped so the expensive
    GPU deploy is shared across this file's tests.
    """
    endpoint_name = random_suffix_name("whisperx-byom", 50)
    model_name = endpoint_name
    role_arn = aws_session.resolve_role_arn(SAGEMAKER_ROLE)

    LOGGER.info(f"Using image: {image_uri}")
    LOGGER.info(f"ModelDataUrl: {MODEL_DATA_URL}")

    model = endpoint_config = endpoint = None
    try:
        # HF_HUB_OFFLINE=1 makes this a no-false-pass test: with the Hub blocked, a
        # served transcription can ONLY come from the ModelDataUrl model staged at
        # /opt/ml/model. Without it, a silent staging failure would fall through to
        # the large-v2 default and download it from the Hub, passing regardless
        # (the response carries no model id to catch that). Offline is safe for
        # startup: VAD ships in the whisperx wheel and diarization loads from the
        # baked local path, so only the Whisper load is gated to the staged model.
        # WHISPERX_DEFAULT_MODEL is still left UNSET so the entrypoint auto-detects
        # /opt/ml/model.
        model = Model.create(
            model_name=model_name,
            primary_container=ContainerDefinition(
                image=image_uri,
                model_data_url=MODEL_DATA_URL,
                environment={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            ),
            execution_role_arn=role_arn,
        )

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


def test_model_data_url_endpoint_serves(model_data_endpoint, audio_en):
    """An offline ModelDataUrl (S3) model deploys and transcribes over /invocations.

    The endpoint runs with HF_HUB_OFFLINE=1 (see the fixture), so reaching InService
    and returning text can only come from the S3 tarball staged to /opt/ml/model,
    auto-detected by the entrypoint — a Hub fallback is impossible. This invocation
    confirms the served endpoint produces a real transcription.
    """
    body = _invoke_transcription(model_data_endpoint, audio_en, language="en")
    text = body.get("text", "")
    assert isinstance(text, str) and text.strip(), (
        f"empty text: the offline ModelDataUrl model did not serve a transcription: {body!r}"
    )
    LOGGER.info(f"ModelDataUrl model served; text (len={len(text)}): {text[:120]!r}")
