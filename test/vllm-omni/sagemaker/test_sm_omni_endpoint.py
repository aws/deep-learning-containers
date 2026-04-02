"""Integration test for vLLM-Omni SageMaker endpoint"""

import json
import logging
from pprint import pformat

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
                "SM_VLLM_ENFORCE_EAGER": "true",
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


@pytest.mark.parametrize("instance_type", ["ml.g4dn.xlarge"], indirect=True)
@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"], indirect=True)
def test_vllm_omni_tts_endpoint(model_endpoint):
    predictor = model_endpoint

    payload = {
        "messages": [{"role": "user", "content": "Hello, this is a test."}],
        "extra_body": {
            "task_type": "CustomVoice",
            "language": "English",
            "speaker": "Ryan",
        },
    }
    LOGGER.info(f"Sending TTS inference request: {pformat(payload)}")

    response = predictor.predict(payload)
    if isinstance(response, bytes):
        response = response.decode("utf-8")
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            pass

    assert response, "Model response is empty"
    LOGGER.info(f"TTS response received: {pformat(response)}")
    assert "choices" in response, f"No choices in response: {response}"
    LOGGER.info("TTS endpoint test PASSED")
