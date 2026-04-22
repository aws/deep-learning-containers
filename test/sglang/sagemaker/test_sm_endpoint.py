"""Integration test for serving endpoint with SGLang DLC — SageMaker SDK v3"""

import json
import logging
from pprint import pformat

import pytest
from sagemaker.serve import ModelBuilder
from test_utils import clean_string, random_suffix_name, wait_for_status
from test_utils.constants import INFERENCE_AMI_VERSION, SAGEMAKER_ROLE
from test_utils.huggingface_helper import get_hf_token

# To enable debugging, change logging.INFO to logging.DEBUG
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

ENDPOINT_WAIT_PERIOD = 60
ENDPOINT_WAIT_LENGTH = 30
ENDPOINT_INSERVICE = "InService"


def get_endpoint_status(sagemaker_client, endpoint_name):
    response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    LOGGER.debug(f"Describe endpoint response: {pformat(response)}")
    return response["EndpointStatus"]


@pytest.fixture(scope="function")
def model_id(request):
    return request.param


@pytest.fixture(scope="function")
def instance_type(request):
    return request.param


@pytest.fixture(scope="function")
def model_endpoint(aws_session, image_uri, model_id, instance_type):
    sagemaker_client = aws_session.sagemaker
    cleaned_id = clean_string(model_id.split("/")[1], "_./")
    endpoint_name = random_suffix_name(f"sglang-{cleaned_id}", 50)

    LOGGER.debug(f"Using image: {image_uri}")
    LOGGER.debug(f"Model ID: {model_id}")

    hf_token = get_hf_token(aws_session)
    model_builder = ModelBuilder(
        image_uri=image_uri,
        role_arn=SAGEMAKER_ROLE,
        env_vars={
            "SM_SGLANG_MODEL_PATH": model_id,
            "HF_TOKEN": hf_token,
        },
    )

    try:
        LOGGER.info(f"Deploying endpoint: {endpoint_name} (this may take 10-15 minutes)...")
        predictor = model_builder.deploy(
            instance_type=instance_type,
            initial_instance_count=1,
            endpoint_name=endpoint_name,
            inference_ami_version=INFERENCE_AMI_VERSION,
        )
        LOGGER.info("Endpoint deployment completed successfully")

        LOGGER.info(f"Waiting for endpoint {ENDPOINT_INSERVICE} status ...")
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
        LOGGER.info(f"Deleting endpoint configuration: {endpoint_name}")
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        # Clean up model via endpoint config
        try:
            ep_cfg = sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_name)
            for v in ep_cfg.get("ProductionVariants", []):
                if v.get("ModelName"):
                    sagemaker_client.delete_model(ModelName=v["ModelName"])
        except Exception:
            pass


@pytest.mark.parametrize("instance_type", ["ml.g5.12xlarge"], indirect=True)
@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-0.6B"], indirect=True)
def test_sglang_sagemaker_endpoint(model_endpoint, model_id):
    predictor = model_endpoint

    prompt = "Write a python script to calculate square of n"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2400,
        "temperature": 0.01,
        "top_p": 0.9,
        "top_k": 50,
    }
    LOGGER.debug(f"Sending inference request with payload: {pformat(payload)}")

    response = predictor.predict(payload)
    LOGGER.info("Inference request invoked successfully")

    if isinstance(response, bytes):
        response = response.decode("utf-8")

    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            LOGGER.warning("Response is not valid JSON. Returning as string.")

    assert response, "Model response is empty, failing endpoint test!"

    LOGGER.info(f"Model response: {pformat(response)}")
    LOGGER.info("Inference test successful!")
