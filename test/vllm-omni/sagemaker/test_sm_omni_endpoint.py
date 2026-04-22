"""Integration test for vLLM-Omni SageMaker endpoint — SageMaker SDK v3"""

import json
import logging
import time

import boto3
import pytest
from test_utils import clean_string, random_suffix_name, wait_for_status
from test_utils.constants import INFERENCE_AMI_VERSION, SAGEMAKER_ROLE
from test_utils.huggingface_helper import get_hf_token

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

ENDPOINT_WAIT_PERIOD = 60
ENDPOINT_WAIT_LENGTH = 30
ENDPOINT_INSERVICE = "InService"


def get_endpoint_status(sagemaker_client, endpoint_name):
    response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    return response["EndpointStatus"]


def _cleanup_resources(sagemaker_client, endpoint_name, model_name):
    """Best-effort cleanup of endpoint, endpoint config, and model."""
    for cleanup_fn, name in [
        (lambda: sagemaker_client.delete_endpoint(EndpointName=endpoint_name), "endpoint"),
        (
            lambda: sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name),
            "endpoint config",
        ),
        (lambda: sagemaker_client.delete_model(ModelName=model_name), "model"),
    ]:
        try:
            cleanup_fn()
        except Exception as e:
            LOGGER.warning(f"Cleanup {name} failed: {e}")


def _create_and_deploy(sagemaker_client, model_name, endpoint_name, image_uri, instance_type, env):
    """Create model, endpoint config, and endpoint. Returns when endpoint is InService."""
    LOGGER.info(f"Creating model: {model_name}")
    sagemaker_client.create_model(
        ModelName=model_name,
        PrimaryContainer={"Image": image_uri, "Environment": env},
        ExecutionRoleArn=SAGEMAKER_ROLE,
    )

    LOGGER.info(f"Creating endpoint config: {endpoint_name}")
    sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": instance_type,
                "InferenceAmiVersion": INFERENCE_AMI_VERSION,
            },
        ],
    )

    LOGGER.info(f"Deploying endpoint: {endpoint_name}")
    sagemaker_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_name,
    )

    LOGGER.info(f"Waiting for endpoint {ENDPOINT_INSERVICE} status...")
    assert wait_for_status(
        ENDPOINT_INSERVICE,
        ENDPOINT_WAIT_PERIOD,
        ENDPOINT_WAIT_LENGTH,
        get_endpoint_status,
        sagemaker_client,
        endpoint_name,
    )


@pytest.fixture(scope="function")
def model_id(request):
    return request.param


@pytest.fixture(scope="function")
def instance_type(request):
    return request.param


@pytest.fixture(scope="function")
def model_endpoint(aws_session, image_uri, model_id, instance_type):
    sagemaker_client = aws_session.sagemaker
    sm_runtime = boto3.client("sagemaker-runtime")
    cleaned_id = clean_string(model_id.split("/")[1], "_./")
    endpoint_name = random_suffix_name(f"vllm-omni-{cleaned_id}", 50)
    model_name = endpoint_name

    hf_token = get_hf_token(aws_session)
    env = {"SM_VLLM_MODEL": model_id, "HF_TOKEN": hf_token}

    try:
        _create_and_deploy(
            sagemaker_client, model_name, endpoint_name, image_uri, instance_type, env
        )
        yield endpoint_name, sm_runtime
    finally:
        _cleanup_resources(sagemaker_client, endpoint_name, model_name)


@pytest.mark.parametrize("instance_type", ["ml.g5.xlarge"], indirect=True)
@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"], indirect=True)
def test_vllm_omni_tts_endpoint(model_endpoint):
    """TTS via /invocations routed to /v1/audio/speech by the serve proxy."""
    endpoint_name, sm_runtime = model_endpoint

    payload = json.dumps(
        {
            "input": "Hello, this is a test of the text to speech system.",
            "voice": "vivian",
            "language": "English",
        }
    )

    LOGGER.info("Sending TTS request via /invocations with route=/v1/audio/speech")
    # First request triggers torch.compile + CUDA graph capture (~67s),
    # which exceeds SageMaker's 60s invoke timeout. Retry after warmup completes.
    for attempt in range(3):
        try:
            response = sm_runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="application/json",
                Body=payload,
                CustomAttributes="route=/v1/audio/speech",
            )
            break
        except Exception as e:
            LOGGER.warning(f"Attempt {attempt + 1}/3 failed: {e}")
            if attempt == 2:
                raise
            time.sleep(30)

    audio_bytes = response["Body"].read()
    LOGGER.info(f"TTS audio response: {len(audio_bytes)} bytes")
    assert len(audio_bytes) > 1000, f"TTS output too small: {len(audio_bytes)} bytes"
    LOGGER.info("TTS endpoint test PASSED")


@pytest.fixture(scope="function")
def async_endpoint(aws_session, image_uri, model_id, instance_type):
    """Deploy an async inference endpoint (no 60s timeout limit)."""
    sagemaker_client = aws_session.sagemaker
    sm_runtime = boto3.client("sagemaker-runtime")
    s3_client = boto3.client("s3")
    cleaned_instance = clean_string(instance_type, "_./")
    endpoint_name = random_suffix_name(f"vllm-omni-async-{cleaned_instance}", 50)
    model_name = endpoint_name
    account_id = aws_session.sts.get_caller_identity()["Account"]
    s3_output = f"s3://sagemaker-{aws_session.region}-{account_id}/vllm-omni-async-output/"

    hf_token = get_hf_token(aws_session)
    env = {"SM_VLLM_MODEL": model_id, "HF_TOKEN": hf_token}

    try:
        LOGGER.info(f"Creating model: {model_name}")
        sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={"Image": image_uri, "Environment": env},
            ExecutionRoleArn=SAGEMAKER_ROLE,
        )

        LOGGER.info(f"Creating async endpoint config: {endpoint_name}")
        sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InitialInstanceCount": 1,
                    "InstanceType": instance_type,
                    "InferenceAmiVersion": INFERENCE_AMI_VERSION,
                },
            ],
            AsyncInferenceConfig={
                "OutputConfig": {"S3OutputPath": s3_output},
                "ClientConfig": {"MaxConcurrentInvocationsPerInstance": 1},
            },
        )

        LOGGER.info(f"Deploying async endpoint: {endpoint_name}")
        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_name,
        )

        LOGGER.info(f"Waiting for endpoint {ENDPOINT_INSERVICE} status...")
        assert wait_for_status(
            ENDPOINT_INSERVICE,
            ENDPOINT_WAIT_PERIOD,
            ENDPOINT_WAIT_LENGTH,
            get_endpoint_status,
            sagemaker_client,
            endpoint_name,
        )
        yield endpoint_name, sm_runtime, s3_client, s3_output
    finally:
        _cleanup_resources(sagemaker_client, endpoint_name, model_name)


@pytest.mark.parametrize("instance_type", ["ml.g5.xlarge"], indirect=True)
@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"], indirect=True)
def test_vllm_omni_tts_async_endpoint(async_endpoint):
    """TTS via async inference — no 60s timeout, up to 1 hour."""
    endpoint_name, sm_runtime, s3_client, s3_output = async_endpoint

    payload = json.dumps(
        {
            "input": "Hello, this is a test of async text to speech.",
            "voice": "vivian",
            "language": "English",
        }
    )

    LOGGER.info("Sending async TTS request")
    response = sm_runtime.invoke_endpoint_async(
        EndpointName=endpoint_name,
        ContentType="application/json",
        InputLocation=_upload_payload_to_s3(s3_client, payload, s3_output, endpoint_name),
        CustomAttributes="route=/v1/audio/speech",
    )

    output_location = response["OutputLocation"]
    LOGGER.info(f"Async output location: {output_location}")

    # Poll for result (up to 5 minutes)
    bucket, key = _parse_s3_uri(output_location)
    for i in range(60):
        try:
            obj = s3_client.get_object(Bucket=bucket, Key=key)
            audio_bytes = obj["Body"].read()
            LOGGER.info(f"Async TTS response: {len(audio_bytes)} bytes (after {i * 5}s)")
            assert len(audio_bytes) > 1000, f"TTS output too small: {len(audio_bytes)} bytes"
            LOGGER.info("Async TTS endpoint test PASSED")
            return
        except s3_client.exceptions.NoSuchKey:
            time.sleep(5)

    pytest.fail("Async inference timed out after 300s")


def _upload_payload_to_s3(s3_client, payload, s3_output, endpoint_name):
    """Upload request payload to S3 for async inference."""
    bucket, prefix = _parse_s3_uri(s3_output)
    key = f"{prefix}{endpoint_name}-input.json"
    s3_client.put_object(Bucket=bucket, Key=key, Body=payload, ContentType="application/json")
    return f"s3://{bucket}/{key}"


def _parse_s3_uri(uri):
    """Parse s3://bucket/key into (bucket, key)."""
    parts = uri.replace("s3://", "").split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""
