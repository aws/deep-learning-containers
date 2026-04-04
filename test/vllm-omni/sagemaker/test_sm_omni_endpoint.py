"""Integration test for vLLM-Omni SageMaker endpoint"""

import json
import logging

import pytest
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
