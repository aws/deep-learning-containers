"""Integration test for serving endpoint with vLLM DLC — SageMaker SDK v3"""

import json
import logging
from pprint import pformat

import pytest
from sagemaker.core.resources import Endpoint, EndpointConfig, Model
from sagemaker.core.shapes import ContainerDefinition, ProductionVariant
from test_utils import clean_string, random_suffix_name
from test_utils.constants import INFERENCE_AMI_VERSION, SAGEMAKER_ROLE
from test_utils.huggingface_helper import get_hf_token
from test_utils.instance_capacity import build_instance_pools

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


def _deploy_endpoint(aws_session, image_uri, model_id, instance_types):
    """Deploy one endpoint over an instance-pool ladder and wait for InService.

    The endpoint carries an instance-pool ladder, so SageMaker walks ``instance_types``
    server-side, provisioning the highest-priority type with capacity and falling back
    to the next on an insufficient-capacity error within this single deploy. Cleanup
    still runs on failure, so a deploy that fails outright does not leak a Model, an
    EndpointConfig, and a Failed Endpoint.
    """
    cleaned_id = clean_string(model_id.split("/")[1], "_./")
    endpoint_name = random_suffix_name(f"vllm-{cleaned_id}", 50)
    model_name = endpoint_name

    hf_token = get_hf_token(aws_session)
    role_arn = aws_session.resolve_role_arn(SAGEMAKER_ROLE)

    model = endpoint_config = endpoint = None
    try:
        LOGGER.info(f"Creating model: {model_name}")
        model = Model.create(
            model_name=model_name,
            primary_container=ContainerDefinition(
                image=image_uri,
                environment={
                    "SM_VLLM_MODEL": model_id,
                    "HF_TOKEN": hf_token,
                },
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
                    instance_pools=build_instance_pools(instance_types),
                    # Give SageMaker room to walk the whole pool ladder on ICE before
                    # failing; the priority-1 pool provisions first when it has capacity.
                    variant_instance_provision_timeout_in_seconds=1800,
                    inference_ami_version=INFERENCE_AMI_VERSION,
                ),
            ],
        )

        LOGGER.info(
            f"Deploying endpoint: {endpoint_name} on {instance_types} "
            f"(this may take 10-15 minutes)..."
        )
        endpoint = Endpoint.create(
            endpoint_name=endpoint_name,
            endpoint_config_name=endpoint_name,
        )
        # Pool provisioning can consume the full 1800s cap on its own, separate from
        # model download and container startup, so the wall-clock wait must cover
        # provisioning plus boot (1800 + ~2700 boot budget = 4500).
        endpoint.wait_for_status("InService", timeout=4500)
    except Exception:
        _cleanup([endpoint, endpoint_config, model])
        raise

    LOGGER.info(f"Endpoint InService: {endpoint_name} on {instance_types}")
    return model, endpoint_config, endpoint


@pytest.fixture(scope="function")
def model_endpoint(aws_session, image_uri, model_id, instance_type):
    """Deploy the endpoint with an instance-pool ladder; SageMaker handles fallback.

    ``instance_type`` may be a single type or a priority-ordered ladder. The ladder
    becomes a production-variant instance pool, so a dry pool in one size falls back to
    the next size up server-side rather than failing. An endpoint that never reaches
    InService fails: skipping would leave the image unvalidated while the run still
    reported green.
    """
    LOGGER.debug(f"Using image: {image_uri}")
    LOGGER.debug(f"Model ID: {model_id}")

    model, endpoint_config, endpoint = _deploy_endpoint(
        aws_session, image_uri, model_id, instance_type
    )
    try:
        yield endpoint
    finally:
        _cleanup([endpoint, endpoint_config, model])


# Ladder, not a single type: all three carry the same single L4 24GB card, so they serve
# this model identically and differ only in capacity pool.
@pytest.mark.parametrize(
    "instance_type",
    [["ml.g6.xlarge", "ml.g6.2xlarge", "ml.g6.4xlarge"]],
    indirect=True,
)
@pytest.mark.parametrize("model_id", ["deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"], indirect=True)
def test_vllm_sagemaker_endpoint(model_endpoint):
    endpoint = model_endpoint

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

    result = endpoint.invoke(body=payload, content_type="application/json")
    body = json.loads(result.body.read())
    LOGGER.info("Inference request invoked successfully")

    assert body, "Model response is empty, failing endpoint test!"

    LOGGER.info(f"Model response: {pformat(body)}")
    LOGGER.info("Inference test successful!")
