"""WhisperX ASR SageMaker async-inference integration test — SageMaker SDK v3.

Async inference removes SageMaker's 60s real-time invoke cap, which is the
recommended path for long audio. The request payload is uploaded to S3, the
endpoint is invoked with `invoke_endpoint_async`, and the JSON response is polled
back from an S3 output location.

Config source: this suite uses the shared `image_uri` pytest fixture + `aws_session`
(the newer pattern already proven for async by test/vllm-omni), NOT openfold3's
TEST_IMAGE_URI/SM_ROLE_ARN env vars — so the CI workflow can pass `--image-uri`
uniformly across the EC2/sync/async suites. The openfold3 async *mechanics*
(S3 preflight + writability probe, stale-resource sweep, failure-location handling,
S3 polling) are mirrored here.

S3 buckets: the SageMaker execution role (AmazonSageMakerFullAccess) can only
read/write buckets whose name contains "sagemaker", so both the async input payload
and the async output live in the account default `sagemaker-<region>-<account>`
bucket. The audio fixture is read from `dlc-cicd-models` with the test runner's own
credentials, then re-uploaded (wrapped in a multipart body) to the sagemaker bucket.

AMI pin (DESIGN decision 9): CUDA 12.8 image -> must use INFERENCE_AMI_VERSION_CU12
(AL2, driver 550). The default CU13 AMI causes a zero-log CannotStartContainerError.

Validation is smoke-level and deterministic: a successful async invocation returns
JSON with non-empty transcription text. WhisperX output is not byte-stable.
"""

import json
import logging
import time
import uuid

import pytest
from sagemaker.core.resources import Endpoint, EndpointConfig, Model
from sagemaker.core.shapes import (
    AsyncInferenceClientConfig,
    AsyncInferenceConfig,
    AsyncInferenceOutputConfig,
    ContainerDefinition,
    ProductionVariant,
)
from test_utils import random_suffix_name
from test_utils.constants import INFERENCE_AMI_VERSION_CU12, SAGEMAKER_ROLE

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

INSTANCE_TYPE = "ml.g6.xlarge"
MAX_CONCURRENT_INVOCATIONS = 1
# Generous: startup warms the Whisper model + diarization pipeline before /ping.
STARTUP_HEALTH_CHECK_TIMEOUT = 1200
DEPLOY_TIMEOUT = 1800

# Source audio fixture (read with the test runner's credentials).
AUDIO_BUCKET = "dlc-cicd-models"
AUDIO_PREFIX = "test-fixtures/audio"
AUDIO_LONG = "asr_long_60s.wav"  # English 60s — long-audio async path

# S3 layout on the sagemaker-named bucket (writable by the SageMaker role).
INPUT_PREFIX = "whisperx-async-input"
OUTPUT_PREFIX = "whisperx-async-output"

# Bounded so a stuck request fails fast instead of hanging.
POLL_TIMEOUT = 900
POLL_INTERVAL = 5

RESOURCE_NAME_PREFIX = "whisperx-async"


def _split(uri: str) -> tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key)."""
    return uri.split("/")[2], "/".join(uri.split("/")[3:])


def _build_multipart(audio_bytes: bytes, filename: str, fields: dict) -> tuple[bytes, str]:
    """Build a multipart/form-data body with the audio `file` part + string fields."""
    boundary = uuid.uuid4().hex
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="{filename}"\r\nContent-Type: audio/wav\r\n\r\n'.encode(),
        audio_bytes,
        b"\r\n",
    ]
    for key, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n'
            f"{value}\r\n".encode()
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


@pytest.fixture(scope="module")
def async_endpoint(aws_session, image_uri):
    """Deploy one async WhisperX endpoint for the module; sweep stale first, clean up after.

    Yields (endpoint_name, io_bucket) — the test uploads its own input payload and
    polls the output, mirroring openfold3's boto3 async flow.
    """
    s3 = aws_session.s3
    account_id = aws_session.sts.get_caller_identity()["Account"]
    io_bucket = f"sagemaker-{aws_session.region}-{account_id}"

    _preflight_s3(s3, io_bucket)
    _sweep_stale(aws_session.sagemaker)

    endpoint_name = random_suffix_name(RESOURCE_NAME_PREFIX, 50)
    model_name = endpoint_name
    role_arn = aws_session.resolve_role_arn(SAGEMAKER_ROLE)
    s3_output = f"s3://{io_bucket}/{OUTPUT_PREFIX}/"

    LOGGER.info(f"Using image: {image_uri}")

    model = endpoint_config = endpoint = None
    try:
        LOGGER.info(f"Creating model: {model_name}")
        model = Model.create(
            model_name=model_name,
            primary_container=ContainerDefinition(image=image_uri),
            execution_role_arn=role_arn,
        )

        LOGGER.info(f"Creating async endpoint config: {endpoint_name}")
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
            async_inference_config=AsyncInferenceConfig(
                output_config=AsyncInferenceOutputConfig(s3_output_path=s3_output),
                client_config=AsyncInferenceClientConfig(
                    max_concurrent_invocations_per_instance=MAX_CONCURRENT_INVOCATIONS,
                ),
            ),
        )

        LOGGER.info(f"Deploying async endpoint {endpoint_name} (~10-15 min)...")
        endpoint = Endpoint.create(endpoint_name=endpoint_name, endpoint_config_name=endpoint_name)
        endpoint.wait_for_status("InService", timeout=DEPLOY_TIMEOUT)
        LOGGER.info("Async endpoint InService")

        yield endpoint_name, io_bucket
    finally:
        for resource in (endpoint, endpoint_config, model):
            if resource is None:
                continue
            try:
                resource.delete()
            except Exception as e:
                LOGGER.warning(f"Cleanup {type(resource).__name__} failed: {e}")


def _preflight_s3(s3, io_bucket: str):
    """Fail before the ~10-min deploy if the fixture is missing or output isn't writable."""
    try:
        s3.head_object(Bucket=AUDIO_BUCKET, Key=f"{AUDIO_PREFIX}/{AUDIO_LONG}")
    except s3.exceptions.ClientError as e:
        raise AssertionError(
            f"Audio fixture s3://{AUDIO_BUCKET}/{AUDIO_PREFIX}/{AUDIO_LONG} not found ({e})."
        ) from e
    probe = f"{OUTPUT_PREFIX}/.preflight-{uuid.uuid4()}"
    try:
        s3.put_object(Bucket=io_bucket, Key=probe, Body=b"ok")
        s3.delete_object(Bucket=io_bucket, Key=probe)
    except s3.exceptions.ClientError as e:
        raise AssertionError(
            f"Cannot write to s3://{io_bucket}/{OUTPUT_PREFIX}/ ({e}). The async endpoint "
            f"output will not be writable — the bucket name must contain 'sagemaker' so the "
            f"SageMaker execution role can write to it."
        ) from e


def _sweep_stale(sm):
    """Delete leftover whisperx-async-* SageMaker resources from a prior canceled run."""
    try:
        for ep in sm.list_endpoints(NameContains=RESOURCE_NAME_PREFIX).get("Endpoints", []):
            LOGGER.warning(f"[sweep] deleting stale endpoint {ep['EndpointName']}")
            try:
                sm.delete_endpoint(EndpointName=ep["EndpointName"])
            except Exception as e:
                LOGGER.warning(f"[sweep] {e}")
        for c in sm.list_endpoint_configs(NameContains=RESOURCE_NAME_PREFIX).get(
            "EndpointConfigs", []
        ):
            try:
                sm.delete_endpoint_config(EndpointConfigName=c["EndpointConfigName"])
            except Exception as e:
                LOGGER.warning(f"[sweep] {e}")
        for m in sm.list_models(NameContains=RESOURCE_NAME_PREFIX).get("Models", []):
            try:
                sm.delete_model(ModelName=m["ModelName"])
            except Exception as e:
                LOGGER.warning(f"[sweep] {e}")
    except Exception as e:
        LOGGER.warning(f"[sweep] skipped: {e}")


def test_async_long_audio(async_endpoint, aws_session):
    """Long (60s) audio transcribes through the async S3-in/S3-out path -> non-empty text."""
    endpoint_name, io_bucket = async_endpoint
    s3 = aws_session.s3
    smr = aws_session.session.client("sagemaker-runtime")

    # Fetch the fixture with the test runner's creds, wrap it in a multipart body, and
    # stage it on the sagemaker bucket (the only bucket the SageMaker role can read).
    audio_bytes = s3.get_object(Bucket=AUDIO_BUCKET, Key=f"{AUDIO_PREFIX}/{AUDIO_LONG}")[
        "Body"
    ].read()
    body, content_type = _build_multipart(audio_bytes, AUDIO_LONG, {"language": "en"})
    input_key = f"{INPUT_PREFIX}/{endpoint_name}-input.bin"
    s3.put_object(Bucket=io_bucket, Key=input_key, Body=body, ContentType=content_type)
    input_location = f"s3://{io_bucket}/{input_key}"

    LOGGER.info(f"Submitting async invocation for {input_location}")
    resp = smr.invoke_endpoint_async(
        EndpointName=endpoint_name,
        InputLocation=input_location,
        ContentType=content_type,
    )
    output_bucket, output_key = _split(resp["OutputLocation"])
    failure_uri = resp.get("FailureLocation", "")
    failure_bucket, failure_key = _split(failure_uri) if failure_uri else (None, None)

    # Poll for the output object; surface a written failure object immediately.
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        try:
            obj = s3.get_object(Bucket=output_bucket, Key=output_key)
            result = json.loads(obj["Body"].read())
            text = result.get("text", "")
            assert isinstance(text, str) and text.strip(), (
                f"Async transcription text is empty: {result!r}"
            )
            LOGGER.info(f"Async transcription text (len={len(text)}): {text[:120]!r}")
            return
        except s3.exceptions.ClientError:
            pass
        if failure_bucket:
            try:
                s3.head_object(Bucket=failure_bucket, Key=failure_key)
                err = (
                    s3.get_object(Bucket=failure_bucket, Key=failure_key)["Body"]
                    .read()
                    .decode()[:2000]
                )
                raise AssertionError(f"SageMaker async failure object: {err}")
            except s3.exceptions.ClientError:
                pass
        time.sleep(POLL_INTERVAL)

    raise TimeoutError(
        f"No async output at s3://{output_bucket}/{output_key} within {POLL_TIMEOUT}s"
    )
