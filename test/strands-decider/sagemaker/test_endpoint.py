"""A strands-decider model deploys as a SageMaker real-time endpoint from ModelDataUrl and answers.

SageMaker extracts the model tarball (the checkpoint folder and base/, the base model) into /opt/ml/model,
runs `docker run IMAGE serve`, and waits for GET /ping, which the image answers only after loading the model
and warming up. HF_HUB_OFFLINE=1 is the image default, so an answer can only come from the staged model.
"""

import json
import logging

import pytest
from sagemaker.core.resources import Endpoint, EndpointConfig, Model
from sagemaker.core.shapes import ContainerDefinition, ProductionVariant
from test_utils import random_suffix_name
from test_utils.constants import INFERENCE_AMI_VERSION_CU12, SAGEMAKER_ROLE
from test_utils.instance_capacity import (
    build_instance_pools,
    log_capacity_placement,
    ml_reservation_config,
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

MODEL = "strands-decider-2B-qwen3.5-v1-2610"
MODEL_DATA_URL = f"s3://dlc-cicd-models/strands-decider-models/{MODEL}.tar.gz"
# SageMaker falls back down this ladder when a type has no capacity. The model needs ~5 GB of GPU memory,
# so every rung fits it: the g6 sizes (one L4), one L40S, one A10G.
INSTANCE_TYPES = [
    "ml.g6.xlarge",
    "ml.g6.2xlarge",
    "ml.g6.4xlarge",
    "ml.g6e.xlarge",
    "ml.g5.2xlarge",
]
# Model load plus the warm-up (about 70 s on the Qwen3.5 models).
STARTUP_HEALTH_CHECK_TIMEOUT = 900
# SageMaker retries a dry pool for a while before it falls back or fails.
DEPLOY_TIMEOUT = 3600

REQUEST = {
    "state": "Help! My payouts have been failing for 3 days and nobody answers my emails!",
    "questions": {
        "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
        "department": {
            "type": "choice",
            "instructions": "Route it.",
            "criteria": {
                "billing": "money, payouts, invoices",
                "technical": "bugs",
                "sales": "pricing",
            },
        },
    },
}


def _cleanup(resources):
    """Best-effort delete: a leaked endpoint keeps billing, so this runs even when a test fails."""
    for resource in resources:
        if resource is None:
            continue
        try:
            resource.delete()
        except Exception as e:
            LOGGER.warning(f"Cleanup {type(resource).__name__} failed: {e}")


@pytest.fixture(scope="module")
def endpoint(aws_session, image_uri):
    name = random_suffix_name("strands-decider", 50)
    model = endpoint_config = endpoint = None
    try:
        model = Model.create(
            model_name=name,
            primary_container=ContainerDefinition(
                image=image_uri,
                model_data_url=MODEL_DATA_URL,
                environment={"DECIDER_CHECKPOINT": f"/opt/ml/model/{MODEL}"},
            ),
            execution_role_arn=aws_session.resolve_role_arn(SAGEMAKER_ROLE),
        )
        endpoint_config = EndpointConfig.create(
            endpoint_config_name=name,
            production_variants=[
                ProductionVariant(
                    variant_name="AllTraffic",
                    model_name=name,
                    initial_instance_count=1,
                    instance_pools=build_instance_pools(INSTANCE_TYPES),
                    capacity_reservation_config=ml_reservation_config(INSTANCE_TYPES),
                    # The torch wheels bring CUDA 12.8: the CU12 (driver 550) AMI.
                    inference_ami_version=INFERENCE_AMI_VERSION_CU12,
                    container_startup_health_check_timeout_in_seconds=STARTUP_HEALTH_CHECK_TIMEOUT,
                ),
            ],
        )
        LOGGER.info(f"Deploying {name} with {image_uri} and {MODEL_DATA_URL}")
        endpoint = Endpoint.create(endpoint_name=name, endpoint_config_name=name)
        endpoint.wait_for_status("InService", timeout=DEPLOY_TIMEOUT)
        log_capacity_placement(endpoint, name)
        yield endpoint
    finally:
        _cleanup([endpoint, endpoint_config, model])


def test_endpoint_answers_from_the_staged_model(endpoint):
    result = endpoint.invoke(body=json.dumps(REQUEST), content_type="application/json")
    body = json.loads(result.body.read())
    LOGGER.info(f"Response: {body}")

    assert body["model"] == MODEL
    answers = body["answers"]
    # An urgent payout failure reads as urgent on every published model (0.86 on this one).
    assert answers["is_urgent"]["noul"] > 0.6
    assert answers["department"]["choice"] in ("billing", "technical")
    assert sum(answers["department"]["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
