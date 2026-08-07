"""Integration test for the llama.cpp SageMaker DLC endpoint — SageMaker SDK v3.

llama.cpp is CPU-only and serves quantized GGUF models. llama-server itself
exposes only /health + OpenAI-compatible routes, so the SageMaker image fronts
it with an nginx proxy that maps GET /ping -> /health and
POST /invocations -> /v1/chat/completions. This test deploys a real endpoint on
a CPU instance and exercises that proxy path end to end.

The model is pulled from HuggingFace at container start via llama-server's
--hf-repo/--hf-file (the entrypoint maps SM_LLAMA_CPP_HF_REPO / SM_LLAMA_CPP_HF_FILE),
so no model artifact needs to be staged in S3.
"""

import json
import logging
from pprint import pformat

import pytest
from sagemaker.core.resources import Endpoint, EndpointConfig, Model
from sagemaker.core.shapes import ContainerDefinition, ProductionVariant
from test_utils import clean_string, random_suffix_name
from test_utils.constants import SAGEMAKER_ROLE

# To enable debugging, change logging.INFO to logging.DEBUG
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


@pytest.fixture(scope="function")
def hf_repo(request):
    return request.param


@pytest.fixture(scope="function")
def hf_file(request):
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
def model_endpoint(aws_session, image_uri, hf_repo, hf_file, instance_type):
    cleaned_id = clean_string(hf_repo.split("/")[-1], "_./")
    endpoint_name = random_suffix_name(f"llama-cpp-{cleaned_id}", 50)
    model_name = endpoint_name

    LOGGER.debug(f"Using image: {image_uri}")
    LOGGER.debug(f"HF repo: {hf_repo}, file: {hf_file}")

    role_arn = aws_session.resolve_role_arn(SAGEMAKER_ROLE)

    model = endpoint_config = endpoint = None
    try:
        LOGGER.info(f"Creating model: {model_name}")
        model = Model.create(
            model_name=model_name,
            primary_container=ContainerDefinition(
                image=image_uri,
                # llama-server pulls the GGUF from HF at startup; no /opt/ml/model
                # artifact is used, so the entrypoint forwards these as
                # --hf-repo / --hf-file.
                environment={
                    "SM_LLAMA_CPP_HF_REPO": hf_repo,
                    "SM_LLAMA_CPP_HF_FILE": hf_file,
                    "SM_LLAMA_CPP_CTX_SIZE": "4096",
                },
            ),
            execution_role_arn=role_arn,
        )

        LOGGER.info(f"Creating endpoint config: {endpoint_name}")
        # CPU instance — no inference_ami_version (that selects a GPU driver AMI).
        endpoint_config = EndpointConfig.create(
            endpoint_config_name=endpoint_name,
            production_variants=[
                ProductionVariant(
                    variant_name="AllTraffic",
                    model_name=model_name,
                    initial_instance_count=1,
                    instance_type=instance_type,
                    container_startup_health_check_timeout_in_seconds=600,
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


@pytest.mark.parametrize("instance_type", ["ml.c6i.2xlarge"], indirect=True)
@pytest.mark.parametrize("hf_file", ["qwen2.5-0.5b-instruct-q4_0.gguf"], indirect=True)
@pytest.mark.parametrize("hf_repo", ["Qwen/Qwen2.5-0.5B-Instruct-GGUF"], indirect=True)
def test_llama_cpp_sagemaker_endpoint(model_endpoint):
    endpoint = model_endpoint

    prompt = "Reply with a single word: hello"
    payload = json.dumps(
        {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16,
            "temperature": 0.01,
        }
    )
    LOGGER.debug(f"Sending inference request with payload: {payload}")

    # Routed through nginx: POST /invocations -> llama-server /v1/chat/completions.
    result = endpoint.invoke(body=payload, content_type="application/json")
    body = json.loads(result.body.read())
    LOGGER.info("Inference request invoked successfully")

    assert body, "Model response is empty, failing endpoint test!"
    # Assert the OpenAI-shaped response made it back through the proxy.
    assert body.get("choices"), f"Response missing 'choices': {pformat(body)}"
    content = body["choices"][0].get("message", {}).get("content")
    assert content, f"Response missing assistant content: {pformat(body)}"

    LOGGER.info(f"Model response: {pformat(body)}")
    LOGGER.info("Inference test successful!")
