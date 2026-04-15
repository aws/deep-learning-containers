"""Integration test for vLLM-Omni SageMaker endpoint"""

import json
import logging
import time

import pytest
from sagemaker.serve import ModelBuilder
from sagemaker.serve.configs import InferenceSpec
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


@pytest.fixture(scope="function")
def model_id(request):
    return request.param


@pytest.fixture(scope="function")
def instance_type(request):
    return request.param


@pytest.fixture(scope="function")
def model_package(aws_session, image_uri, model_id):
    sagemaker_client = aws_session.sagemaker
    cleaned_id = clean_string(model_id.split("/")[1], "_./")
    model_name = random_suffix_name(f"vllm-omni-{cleaned_id}", 50)

    try:
        LOGGER.info(f"Creating SageMaker model: {model_name}")
        hf_token = get_hf_token(aws_session)
        inference_spec = InferenceSpec(
            image_uri=image_uri,
            environment={
                "SM_VLLM_MODEL": model_id,
                "HF_TOKEN": hf_token,
            },
        )
        builder = ModelBuilder(
            inference_spec=inference_spec,
            role=SAGEMAKER_ROLE,
        )
        yield builder, model_name
    finally:
        LOGGER.info(f"Deleting model: {model_name}")
        sagemaker_client.delete_model(ModelName=model_name)


@pytest.fixture(scope="function")
def model_endpoint(aws_session, model_package, instance_type):
    sagemaker_client = aws_session.sagemaker
    builder, _ = model_package
    cleaned_instance = clean_string(instance_type, "_./")
    endpoint_name = random_suffix_name(f"vllm-omni-{cleaned_instance}", 50)

    try:
        LOGGER.info("Starting endpoint deployment...")
        endpoint = builder.deploy(
            instance_type=instance_type,
            initial_instance_count=1,
            endpoint_name=endpoint_name,
            inference_ami_version=INFERENCE_AMI_VERSION,
            wait=True,
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
        yield endpoint, endpoint_name
    finally:
        LOGGER.info(f"Deleting endpoint: {endpoint_name}")
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)


@pytest.mark.parametrize("instance_type", ["ml.g5.xlarge"], indirect=True)
@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"], indirect=True)
def test_vllm_omni_tts_endpoint(model_endpoint, aws_session):
    """TTS via /invocations routed to /v1/audio/speech by the serve proxy."""
    endpoint, endpoint_name = model_endpoint
    sm_runtime = aws_session.session.client("sagemaker-runtime")

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
def async_endpoint(aws_session, model_package, instance_type):
    """Deploy an async inference endpoint (no 60s timeout limit)."""
    sagemaker_client = aws_session.sagemaker
    builder, _ = model_package
    cleaned_instance = clean_string(instance_type, "_./")
    endpoint_name = random_suffix_name(f"vllm-omni-async-{cleaned_instance}", 50)
    account_id = aws_session.sts.get_caller_identity()["Account"]
    s3_output = f"s3://sagemaker-{aws_session.region}-{account_id}/vllm-omni-async-output/"

    try:
        LOGGER.info(f"Deploying async endpoint: {endpoint_name}")
        # For async inference in V3, use boto3 directly to create the async endpoint config
        # since ModelBuilder.deploy() handles standard real-time endpoints
        sm = aws_session.sagemaker

        # First build the model
        endpoint = builder.deploy(
            instance_type=instance_type,
            initial_instance_count=1,
            endpoint_name=endpoint_name,
            inference_ami_version=INFERENCE_AMI_VERSION,
            async_inference_config={
                "OutputConfig": {
                    "S3OutputPath": s3_output,
                },
                "ClientConfig": {
                    "MaxConcurrentInvocationsPerInstance": 1,
                },
            },
            wait=True,
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
        yield endpoint, endpoint_name, s3_output
    finally:
        LOGGER.info(f"Deleting async endpoint: {endpoint_name}")
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)


@pytest.mark.parametrize("instance_type", ["ml.g5.xlarge"], indirect=True)
@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"], indirect=True)
def test_vllm_omni_tts_async_endpoint(async_endpoint, aws_session):
    """TTS via async inference — no 60s timeout, up to 1 hour."""
    endpoint, endpoint_name, s3_output = async_endpoint
    sm_runtime = aws_session.session.client("sagemaker-runtime")
    s3_client = aws_session.s3

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
