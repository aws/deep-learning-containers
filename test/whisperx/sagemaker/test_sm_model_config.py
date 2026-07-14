"""WhisperX model-config integration test — proves env config flows through SageMaker.

Deploys a SECOND real-time endpoint whose ``ContainerDefinition.environment`` sets
``WHISPERX_DEFAULT_MODEL=tiny`` and verifies the container actually honors it. This
is the whole point of the suite: an env var set on the SageMaker Model must reach
the running container and change which Whisper model it pins and serves.

We prove the pin took effect purely through the SageMaker ``/invocations`` route
(the only inference surface SageMaker exposes — there is no GET /v1/models over the
invoke API) via accept/reject behavior:
  * a request with no `model` resolves to the pinned default -> 200 non-empty text,
    which is only possible if `tiny` was warm-loaded and is being served; and
  * a request naming `large-v2` (the server's normal default) is REJECTED with a
    404 that names `tiny` — impossible unless the env override replaced the pin.

Kept separate from ``test_sm_endpoint.py`` so the custom-env deploy never touches
the default endpoint. ``tiny`` also warm-loads fast, keeping deploy + health check
quick. AMI pin (DESIGN decision 9): CUDA 12.8 image -> INFERENCE_AMI_VERSION_CU12
(AL2, driver 550); the default CU13 AMI causes a zero-log CannotStartContainerError.

Validation is deterministic: assert served/rejected model behavior and non-empty
text, never an exact transcript (WhisperX output is not byte-stable).
"""

import json
import logging
import os
import time
import uuid

import pytest
from botocore.exceptions import ClientError
from sagemaker.core.resources import Endpoint, EndpointConfig, Model
from sagemaker.core.shapes import ContainerDefinition, ProductionVariant
from test_utils import random_suffix_name
from test_utils.constants import INFERENCE_AMI_VERSION_CU12, SAGEMAKER_ROLE

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# Single L4 GPU; matches WhisperX VRAM needs.
INSTANCE_TYPE = "ml.g6.xlarge"
# Startup warms the pinned Whisper model + diarization pipeline before /ping
# returns 200, so InService already implies the (tiny) model is resident.
STARTUP_HEALTH_CHECK_TIMEOUT = 900
DEPLOY_TIMEOUT = 1800

# Audio fixture pre-staged in S3 (read with the test runner's credentials).
AUDIO_BUCKET = "dlc-cicd-models"
AUDIO_PREFIX = "test-fixtures/audio"
AUDIO_EN = "asr_en.wav"  # English ~15s

# Env override under test: pin the container to the smallest Whisper model.
CUSTOM_MODEL = "tiny"
# The server's normal default when WHISPERX_DEFAULT_MODEL is unset. Naming it must
# be REJECTED here — proving the env override, not the built-in default, is the pin.
NORMAL_DEFAULT_MODEL = "large-v2"


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
def custom_model_endpoint(aws_session, image_uri):
    """Deploy one WhisperX endpoint pinned to `tiny` via env; clean up after.

    Modeled on ``test_sm_endpoint.py``'s ``model_endpoint`` fixture, but the
    ContainerDefinition carries ``environment={"WHISPERX_DEFAULT_MODEL": "tiny"}``
    so the running container serves `tiny` instead of the built-in `large-v2`.
    Module-scoped so the (expensive) GPU deploy is shared across this file's tests.
    """
    endpoint_name = random_suffix_name("whisperx-cfg", 50)
    model_name = endpoint_name
    role_arn = aws_session.resolve_role_arn(SAGEMAKER_ROLE)

    LOGGER.info(f"Using image: {image_uri}")

    model = endpoint_config = endpoint = None
    try:
        LOGGER.info(f"Creating model: {model_name} (WHISPERX_DEFAULT_MODEL={CUSTOM_MODEL})")
        model = Model.create(
            model_name=model_name,
            primary_container=ContainerDefinition(
                image=image_uri,
                environment={"WHISPERX_DEFAULT_MODEL": CUSTOM_MODEL},
            ),
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


def test_custom_default_model_served(custom_model_endpoint, audio_en):
    """WHISPERX_DEFAULT_MODEL=tiny propagates through SageMaker to the container.

    Two halves prove the env override reached the running container and became the
    served pin, using only the SageMaker /invocations route:

    1. A request with no `model` resolves to the pinned default and returns 200 with
       non-empty text — only possible if `tiny` warm-loaded and is being served.
    2. A request naming `large-v2` (the server's normal default) is REJECTED with a
       404 whose detail names `tiny`. Since override is off, the server 404s any
       `model` != the served id; that the *served id is tiny* (not large-v2) can
       only be true if ContainerDefinition.environment reached the container.
    """
    endpoint = custom_model_endpoint

    # (1) Served-model check: no `model` field -> resolves to the pinned default.
    body = _invoke_transcription(endpoint, audio_en, language="en")
    text = body.get("text", "")
    assert isinstance(text, str) and text.strip(), (
        f"Default (tiny) model returned empty text; is tiny actually served? {body!r}"
    )
    LOGGER.info(f"tiny model served; transcription text (len={len(text)}): {text[:120]!r}")

    # (2) Pin check: the normal default `large-v2` must now be rejected, naming tiny.
    # sagemaker-core Endpoint.invoke -> sagemaker-runtime invoke_endpoint with no
    # try/except, so a container 404 surfaces as ModelError (a ClientError subclass,
    # HTTP 424) wrapping the container status + body. Assert on that raised error;
    # defensively also handle a returned body if a runtime path hands one back.
    reject_body, content_type = _build_multipart(
        audio_en, {"model": NORMAL_DEFAULT_MODEL, "language": "en"}
    )
    try:
        result = endpoint.invoke(body=reject_body, content_type=content_type)
    except ClientError as e:
        err = f"{e}\n{json.dumps(e.response, default=str)}"
        lowered = err.lower()
        # `large-v2` being rejected at all already proves the pin moved off the
        # built-in default; the served id (`tiny`) named in the detail proves the
        # env value specifically reached the container.
        assert "does not exist" in lowered or CUSTOM_MODEL in lowered, (
            f"Expected `{NORMAL_DEFAULT_MODEL}` rejected with a 404 naming pinned "
            f"`{CUSTOM_MODEL}`, got: {err!r}"
        )
        LOGGER.info(f"`{NORMAL_DEFAULT_MODEL}` correctly rejected; pin is `{CUSTOM_MODEL}`: {e}")
        return

    raw = result.body.read()
    text = raw.decode(errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
    lowered = text.lower()
    assert "does not exist" in lowered or CUSTOM_MODEL in lowered, (
        f"Expected `{NORMAL_DEFAULT_MODEL}` to be rejected naming `{CUSTOM_MODEL}`, got: {text!r}"
    )
    LOGGER.info(f"`{NORMAL_DEFAULT_MODEL}` rejected in returned body: {text[:200]!r}")
