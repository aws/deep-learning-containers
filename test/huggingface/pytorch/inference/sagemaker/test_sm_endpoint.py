"""SageMaker endpoint integration tests for Hugging Face PyTorch inference DLC."""

import base64
import json
import logging
import os
from pprint import pformat

import boto3
import pytest
from sagemaker.core.resources import Endpoint, EndpointConfig, Model
from sagemaker.core.shapes import ContainerDefinition, ProductionVariant
from test_utils import clean_string, random_suffix_name
from test_utils.constants import INFERENCE_AMI_VERSION, SAGEMAKER_ROLE
from test_utils.huggingface_helper import get_hf_token

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

ASR_SAMPLE_URL = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"
STARTUP_HEALTH_CHECK_TIMEOUT = 3600
ENDPOINT_WAIT_TIMEOUT = 4200
INSTANCE_TYPE = "ml.g6.xlarge"


@pytest.fixture(scope="function")
def model_config(request):
    return request.param


def _get_hf_token(aws_session):
    token = os.getenv("HF_TOKEN")
    if token:
        return token
    return get_hf_token(aws_session)


def _cleanup(resources):
    for resource in resources:
        if resource is None:
            continue
        try:
            resource.delete()
        except Exception as e:
            LOGGER.warning(f"Cleanup {type(resource).__name__} failed: {e}")


def _build_production_variant(model_name):
    return ProductionVariant(
        variant_name="AllTraffic",
        model_name=model_name,
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        inference_ami_version=INFERENCE_AMI_VERSION,
        container_startup_health_check_timeout_in_seconds=STARTUP_HEALTH_CHECK_TIMEOUT,
    )


def _wait_for_endpoint(endpoint):
    try:
        endpoint.wait_for_status("InService", timeout=ENDPOINT_WAIT_TIMEOUT)
    except Exception as e:
        LOGGER.error("Endpoint deployment failed: %s", e)
        raise


@pytest.fixture(scope="function")
def model_endpoint(aws_session, image_uri, model_config):
    model_id = model_config["model_id"]
    cleaned_id = clean_string(model_id.split("/")[-1], "_./")
    model_name = random_suffix_name(f"hfpt-{cleaned_id}", 50)
    endpoint_name = random_suffix_name(f"hfpt-{cleaned_id}", 50)

    LOGGER.info(f"Using image: {image_uri}")
    LOGGER.info(f"Model ID: {model_id}")

    hf_token = _get_hf_token(aws_session)
    role_arn = aws_session.resolve_role_arn(SAGEMAKER_ROLE)
    env = {
        "MODEL_ID": model_id,
        "TASK": model_config["task"],
        "HF_TOKEN": hf_token,
    }
    env.update(model_config.get("env", {}))

    model = endpoint_config = endpoint = None
    try:
        LOGGER.info(f"Creating model: {model_name}")
        model = Model.create(
            model_name=model_name,
            primary_container=ContainerDefinition(
                image=image_uri,
                environment=env,
            ),
            execution_role_arn=role_arn,
        )

        LOGGER.info(f"Creating endpoint config: {endpoint_name}")
        endpoint_config = EndpointConfig.create(
            endpoint_config_name=endpoint_name,
            production_variants=[_build_production_variant(model_name)],
        )

        LOGGER.info(f"Deploying endpoint: {endpoint_name}")
        endpoint = Endpoint.create(
            endpoint_name=endpoint_name,
            endpoint_config_name=endpoint_name,
        )
        _wait_for_endpoint(endpoint)
        LOGGER.info("Endpoint deployment completed successfully")

        yield {"name": endpoint_name, "config": model_config}
    finally:
        _cleanup([endpoint, endpoint_config, model])


@pytest.mark.parametrize(
    "model_config",
    [
        {
            "model_id": "LiquidAI/LFM2.5-230M",
            "task": "text-generation",
            "env": {"DTYPE": "bfloat16"},
        },
    ],
    indirect=True,
)
def test_text_generation_endpoint(model_endpoint):
    endpoint_name = model_endpoint["name"]
    runtime = boto3.client("sagemaker-runtime")

    payload = {
        "model": "LiquidAI/LFM2.5-230M",
        "messages": [
            {"role": "system", "content": "You answer with a short factual sentence."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        "max_tokens": 32,
        "temperature": 0.5,
    }
    LOGGER.info(f"Sending chat-completion payload: {payload}")

    result = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=json.dumps(payload),
        ContentType="application/json",
        Accept="application/json",
        CustomAttributes="route=/v1/chat/completions",
    )
    body = json.loads(result["Body"].read())
    LOGGER.info(f"Model response: {pformat(body)}")

    content = body["choices"][0]["message"]["content"]
    assert content.strip(), "Generated chat message is empty"


@pytest.mark.parametrize(
    "model_config",
    [
        {
            "model_id": "nvidia/parakeet-tdt-0.6b-v3",
            "task": "automatic-speech-recognition",
            "env": {"DTYPE": "bfloat16"},
        },
    ],
    indirect=True,
)
def test_asr_endpoint(model_endpoint):
    endpoint_name = model_endpoint["name"]
    runtime = boto3.client("sagemaker-runtime")

    # Payload should be a base64 encoded audio file
    with open(ASR_SAMPLE_URL, "rb") as f:
        audio_data = f.read()
    base64_audio_data = base64.b64encode(audio_data).decode("utf-8")
    payload = {"inputs": base64_audio_data}

    LOGGER.info("Sending ASR payload")

    result = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=json.dumps(payload),
        ContentType="application/json",
        Accept="application/json",
        CustomAttributes="route=/predict-json",
    )
    body = json.loads(result["Body"].read())
    LOGGER.info(f"Model response: {pformat(body)}")

    assert body.get("text", "").strip(), "Transcription text is empty"
