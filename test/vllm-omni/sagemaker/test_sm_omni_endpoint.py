"""Integration test for vLLM-Omni SageMaker endpoint"""

import json
import logging
import time

import pytest
from sagemaker.async_inference import AsyncInferenceConfig
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
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
        model = Model(
            name=model_name,
            image_uri=image_uri,
            role=SAGEMAKER_ROLE,
            predictor_cls=Predictor,
            env={
                "SM_VLLM_MODEL": model_id,
                "HF_TOKEN": hf_token,
            },
        )
        yield model
    finally:
        LOGGER.info(f"Deleting model: {model_name}")
        sagemaker_client.delete_model(ModelName=model_name)


@pytest.fixture(scope="function")
def model_endpoint(aws_session, model_package, instance_type):
    sagemaker_client = aws_session.sagemaker
    model = model_package
    cleaned_instance = clean_string(instance_type, "_./")
    endpoint_name = random_suffix_name(f"vllm-omni-{cleaned_instance}", 50)

    try:
        LOGGER.info("Starting endpoint deployment...")
        predictor = model.deploy(
            instance_type=instance_type,
            initial_instance_count=1,
            endpoint_name=endpoint_name,
            inference_ami_version=INFERENCE_AMI_VERSION,
            serializer=JSONSerializer(),
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
        yield predictor
    finally:
        LOGGER.info(f"Deleting endpoint: {endpoint_name}")
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)


@pytest.mark.parametrize("instance_type", ["ml.g5.xlarge"], indirect=True)
@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"], indirect=True)
def test_vllm_omni_tts_endpoint(model_endpoint):
    """TTS via /invocations routed to /v1/audio/speech by the serve proxy."""
    predictor = model_endpoint
    sm_runtime = predictor.sagemaker_session.sagemaker_runtime_client

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
    import time

    # https://github.com/aws/sagemaker-python-sdk/issues/1119
    for attempt in range(3):
        try:
            response = sm_runtime.invoke_endpoint(
                EndpointName=predictor.endpoint_name,
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
    model = model_package
    cleaned_instance = clean_string(instance_type, "_./")
    endpoint_name = random_suffix_name(f"vllm-omni-async-{cleaned_instance}", 50)
    account_id = aws_session.sts.get_caller_identity()["Account"]
    s3_output = f"s3://sagemaker-{aws_session.region}-{account_id}/vllm-omni-async-output/"

    try:
        LOGGER.info(f"Deploying async endpoint: {endpoint_name}")
        predictor = model.deploy(
            instance_type=instance_type,
            initial_instance_count=1,
            endpoint_name=endpoint_name,
            inference_ami_version=INFERENCE_AMI_VERSION,
            serializer=JSONSerializer(),
            async_inference_config=AsyncInferenceConfig(
                output_path=s3_output,
                max_concurrent_invocations_per_instance=1,
            ),
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
        yield predictor, s3_output
    finally:
        LOGGER.info(f"Deleting async endpoint: {endpoint_name}")
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)


@pytest.mark.parametrize("instance_type", ["ml.g5.xlarge"], indirect=True)
@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"], indirect=True)
def test_vllm_omni_tts_async_endpoint(async_endpoint):
    """TTS via async inference — no 60s timeout, up to 1 hour."""
    predictor, s3_output = async_endpoint
    sm_runtime = predictor.sagemaker_session.sagemaker_runtime_client
    s3_client = predictor.sagemaker_session.boto_session.client("s3")

    payload = json.dumps(
        {
            "input": "Hello, this is a test of async text to speech.",
            "voice": "vivian",
            "language": "English",
        }
    )

    LOGGER.info("Sending async TTS request")
    response = sm_runtime.invoke_endpoint_async(
        EndpointName=predictor.endpoint_name,
        ContentType="application/json",
        InputLocation=_upload_payload_to_s3(s3_client, payload, s3_output, predictor.endpoint_name),
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


# ---------------------------------------------------------------------------
# Video generation (Wan2.1) — async endpoint with JSON→form-data middleware
# ---------------------------------------------------------------------------

S3_MODEL_BUCKET = "dlc-cicd-models"
S3_MODEL_PREFIX = "omni-models"


@pytest.fixture(scope="function")
def video_model_package(aws_session, image_uri):
    """Model backed by S3 tarball (not HF download)."""
    sagemaker_client = aws_session.sagemaker
    model_name = random_suffix_name("vllm-omni-video", 50)
    model_data_url = f"s3://{S3_MODEL_BUCKET}/{S3_MODEL_PREFIX}/wan2.1-t2v-1.3b.tar.gz"

    try:
        LOGGER.info(f"Creating video model: {model_name}")
        model = Model(
            name=model_name,
            image_uri=image_uri,
            role=SAGEMAKER_ROLE,
            predictor_cls=Predictor,
            model_data=model_data_url,
            env={
                "SM_VLLM_MAX_MODEL_LEN": "2048",
                "SM_VLLM_ENFORCE_EAGER": "true",
                "SM_VLLM_TENSOR_PARALLEL_SIZE": "2",
            },
        )
        yield model
    finally:
        LOGGER.info(f"Deleting model: {model_name}")
        sagemaker_client.delete_model(ModelName=model_name)


@pytest.fixture(scope="function")
def video_async_endpoint(aws_session, video_model_package):
    """Deploy async endpoint for video generation."""
    sagemaker_client = aws_session.sagemaker
    model = video_model_package
    endpoint_name = random_suffix_name("vllm-omni-async-video", 50)
    account_id = aws_session.sts.get_caller_identity()["Account"]
    s3_output = f"s3://sagemaker-{aws_session.region}-{account_id}/vllm-omni-async-output/"

    try:
        LOGGER.info(f"Deploying video async endpoint: {endpoint_name}")
        predictor = model.deploy(
            instance_type="ml.g5.12xlarge",
            initial_instance_count=1,
            endpoint_name=endpoint_name,
            inference_ami_version=INFERENCE_AMI_VERSION,
            serializer=JSONSerializer(),
            async_inference_config=AsyncInferenceConfig(
                output_path=s3_output,
                max_concurrent_invocations_per_instance=1,
            ),
            wait=True,
            container_startup_health_check_timeout=900,
            model_data_download_timeout=1200,
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
        yield predictor, s3_output
    finally:
        LOGGER.info(f"Deleting video endpoint: {endpoint_name}")
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)


def test_vllm_omni_video_async_endpoint(video_async_endpoint):
    """Video generation via async inference — JSON payload converted to form-data by middleware."""
    predictor, s3_output = video_async_endpoint
    sm_runtime = predictor.sagemaker_session.sagemaker_runtime_client
    s3_client = predictor.sagemaker_session.boto_session.client("s3")

    # JSON payload — middleware converts to multipart/form-data for /v1/videos
    payload = json.dumps(
        {
            "prompt": "a dog running on a beach",
            "num_frames": "17",
            "size": "480x320",
            "seed": "42",
        }
    )

    LOGGER.info("Sending async video request (JSON, middleware converts to form-data)")
    response = sm_runtime.invoke_endpoint_async(
        EndpointName=predictor.endpoint_name,
        ContentType="application/json",
        InputLocation=_upload_payload_to_s3(s3_client, payload, s3_output, predictor.endpoint_name),
        CustomAttributes="route=/v1/videos",
    )

    output_location = response["OutputLocation"]
    LOGGER.info(f"Async output location: {output_location}")

    # Poll for result (up to 10 minutes — video generation is slow)
    bucket, key = _parse_s3_uri(output_location)
    for i in range(120):
        try:
            obj = s3_client.get_object(Bucket=bucket, Key=key)
            body = obj["Body"].read()
            LOGGER.info(f"Async video response: {len(body)} bytes (after {i * 5}s)")
            result = json.loads(body)
            assert "id" in result, f"Expected 'id' in video response, got: {list(result.keys())}"
            LOGGER.info(f"Video job created: {result['id']} (status: {result.get('status')})")
            LOGGER.info("Video async endpoint test PASSED")
            return
        except s3_client.exceptions.NoSuchKey:
            time.sleep(5)

    pytest.fail("Async video inference timed out after 600s")
