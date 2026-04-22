"""Integration test for serving endpoint with vLLM DLC — SageMaker SDK v3"""

import json
import logging
from pprint import pformat

import boto3
import pytest
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
    sm_runtime = boto3.client("sagemaker-runtime")
    cleaned_id = clean_string(model_id.split("/")[1], "_./")
    endpoint_name = random_suffix_name(f"vllm-{cleaned_id}", 50)
    model_name = endpoint_name

    LOGGER.debug(f"Using image: {image_uri}")
    LOGGER.debug(f"Model ID: {model_id}")

    hf_token = get_hf_token(aws_session)

    try:
        LOGGER.info(f"Creating model: {model_name}")
        sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image": image_uri,
                "Environment": {
                    "SM_VLLM_MODEL": model_id,
                    "HF_TOKEN": hf_token,
                },
            },
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

        LOGGER.info(f"Deploying endpoint: {endpoint_name} (this may take 10-15 minutes)...")
        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_name,
        )

        LOGGER.info(f"Waiting for endpoint {ENDPOINT_INSERVICE} status ...")
        assert wait_for_status(
            ENDPOINT_INSERVICE,
            ENDPOINT_WAIT_PERIOD,
            ENDPOINT_WAIT_LENGTH,
            get_endpoint_status,
            sagemaker_client,
            endpoint_name,
        )
        LOGGER.info("Endpoint deployment completed successfully")

        yield endpoint_name, sm_runtime
    finally:
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


@pytest.mark.parametrize("instance_type", ["ml.g5.12xlarge"], indirect=True)
@pytest.mark.parametrize("model_id", ["deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"], indirect=True)
def test_vllm_sagemaker_endpoint(model_endpoint):
    endpoint_name, sm_runtime = model_endpoint

    prompt = "Write a python script to calculate square of n"
    payload = json.dumps(
        {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2400,
            "temperature": 0.01,
            "top_p": 0.9,
            "top_k": 50,
        }
    )
    LOGGER.debug(f"Sending inference request with payload: {payload}")

    response = sm_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=payload,
    )
    body = json.loads(response["Body"].read())
    LOGGER.info("Inference request invoked successfully")

    assert body, "Model response is empty, failing endpoint test!"

    LOGGER.info(f"Model response: {pformat(body)}")
    LOGGER.info("Inference test successful!")
