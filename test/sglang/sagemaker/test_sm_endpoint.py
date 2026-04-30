"""Integration test for serving endpoint with SGLang DLC — SageMaker SDK v3"""

import json
import logging
from pprint import pformat

import pytest
from sagemaker.core.resources import Endpoint, EndpointConfig, Model
from sagemaker.core.shapes import ContainerDefinition, ProductionVariant
from test_utils import clean_string, random_suffix_name
from test_utils.constants import INFERENCE_AMI_VERSION, SAGEMAKER_ROLE
from test_utils.huggingface_helper import get_hf_token

# To enable debugging, change logging.INFO to logging.DEBUG
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


@pytest.fixture(scope="function")
def model_id(request):
    return request.param


@pytest.fixture(scope="function")
def instance_type(request):
    return request.param


def _cleanup(resources):
    """Best-effort delete for a list of v3 resource objects (None-safe)."""
    for resource in resources:
        if resource is None:
            continue
        try:
            resource.delete()
        except Exception as e:
            LOGGER.warning(f"Cleanup {type(resource).__name__} failed: {e}")


@pytest.fixture(scope="function")
def model_endpoint(aws_session, image_uri, model_id, instance_type):
    cleaned_id = clean_string(model_id.split("/")[1], "_./")
    endpoint_name = random_suffix_name(f"sglang-{cleaned_id}", 50)
    model_name = endpoint_name

    LOGGER.debug(f"Using image: {image_uri}")
    LOGGER.debug(f"Model ID: {model_id}")

    hf_token = get_hf_token(aws_session)

    model = endpoint_config = endpoint = None
    try:
        LOGGER.info(f"Creating model: {model_name}")
        model = Model.create(
            model_name=model_name,
            primary_container=ContainerDefinition(
                image=image_uri,
                environment={
                    "SM_SGLANG_MODEL_PATH": model_id,
                    "HF_TOKEN": hf_token,
                },
            ),
            execution_role_arn=SAGEMAKER_ROLE,
        )

        LOGGER.info(f"Creating endpoint config: {endpoint_name}")
        endpoint_config = EndpointConfig.create(
            endpoint_config_name=endpoint_name,
            production_variants=[
                ProductionVariant(
                    variant_name="AllTraffic",
                    model_name=model_name,
                    initial_instance_count=1,
                    instance_type=instance_type,
                    inference_ami_version=INFERENCE_AMI_VERSION,
                ),
            ],
        )

        LOGGER.info(f"Deploying endpoint: {endpoint_name} (this may take 10-15 minutes)...")
        endpoint = Endpoint.create(
            endpoint_name=endpoint_name,
            endpoint_config_name=endpoint_name,
        )
        endpoint.wait_for_status("InService")
        LOGGER.info("Endpoint deployment completed successfully")

        yield endpoint
    finally:
        _cleanup([endpoint, endpoint_config, model])


@pytest.mark.parametrize("instance_type", ["ml.g5.12xlarge"], indirect=True)
@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-0.6B"], indirect=True)
def test_sglang_sagemaker_endpoint(model_endpoint, model_id):
    endpoint = model_endpoint

    prompt = "Write a python script to calculate square of n"
    payload = json.dumps(
        {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2400,
            "temperature": 0.01,
            "top_p": 0.9,
            "top_k": 50,
        }
    )
    LOGGER.debug(f"Sending inference request with payload: {payload}")

    result = endpoint.invoke(body=payload, content_type="application/json")
    body = json.loads(result.body.read())
    LOGGER.info("Inference request invoked successfully")

    assert body, "Model response is empty, failing endpoint test!"

    LOGGER.info(f"Model response: {pformat(body)}")
    LOGGER.info("Inference test successful!")
