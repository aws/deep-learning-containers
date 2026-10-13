"""Config-driven SageMaker endpoint test for vLLM DLC models.

Supports text, audio, and video models. Reads model configuration from
.github/config/model-tests/vllm-sagemaker-endpoint-tests.yml (sagemaker section)
and runs its ``instance_type:`` entries on real SageMaker endpoints. ``fleet:``
entries run as a local container instead, in test_sm_local_serving.py.

Each entry deploys a model from S3 with configured env vars, sends
requests to the specified route, and validates the response.

Filtering by image (e.g. ``required_image_pattern: {os_version: amzn2023}``)
happens in the workflow preflight step via ``scripts/ci/parse_model_config.py``;
the workflow gates the whole endpoint-test job on the resulting matrix being
non-empty. This test runs every ``instance_type:`` entry in the config file, or
only the one named by the ``SM_MODEL_NAME`` environment variable.

Usage:
    SM_MODEL_NAME=<name> pytest test_sm_model_serving.py --image-uri <ecr_uri>
    pytest test_sm_model_serving.py --image-uri <ecr_uri>  # runs all instance_type entries
"""

import json
import logging
import os
import time

import boto3
import pytest
from sagemaker.core.resources import Endpoint, EndpointConfig, Model
from sagemaker.core.shapes import (
    ContainerDefinition,
    ModelDataSource,
    ProductionVariant,
    S3ModelDataSource,
)
from sm_serving_common import (
    build_json_body,
    build_multipart_body,
    container_env,
    load_sagemaker_config,
    validate_response,
)
from test_utils import random_suffix_name
from test_utils.constants import INFERENCE_AMI_VERSION, SAGEMAKER_ROLE
from test_utils.instance_capacity import (
    build_instance_pools,
    normalize_instance_types,
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def _cleanup(resources):
    for resource in resources:
        if resource is None:
            continue
        try:
            resource.delete()
        except Exception as e:
            LOGGER.warning(f"Cleanup {type(resource).__name__} failed: {e}")


def _get_role_arn(region):
    try:
        iam = boto3.client("iam", region_name=region)
        role_info = iam.get_role(RoleName=SAGEMAKER_ROLE)
        return role_info["Role"]["Arn"]
    except Exception:
        sts = boto3.client("sts", region_name=region)
        account = sts.get_caller_identity()["Account"]
        return f"arn:aws:iam::{account}:role/{SAGEMAKER_ROLE}"


def _deploy_endpoint(image_uri, model_cfg, region, instance_types):
    """Deploy one endpoint over an instance-pool ladder and wait for InService.

    The endpoint carries an instance-pool ladder, so SageMaker walks ``instance_types``
    server-side, provisioning the highest-priority type with capacity and falling back
    to the next on an insufficient-capacity error within this single deploy. Cleanup
    still runs on failure, so a deploy that fails outright does not leak a Model, an
    EndpointConfig, and a Failed Endpoint.
    """
    endpoint_name = random_suffix_name(f"vllm-{model_cfg['name']}", 50)
    role_arn = _get_role_arn(region)
    env_vars = container_env(model_cfg)

    model = endpoint_config = endpoint = None
    try:
        LOGGER.info(f"Creating model: {endpoint_name}")
        create_kwargs = dict(
            model_name=endpoint_name,
            primary_container=ContainerDefinition(
                image=image_uri,
                model_data_source=ModelDataSource(
                    s3_data_source=S3ModelDataSource(
                        s3_uri=model_cfg["s3_path"],
                        s3_data_type="S3Prefix",
                        compression_type="Gzip",
                    ),
                ),
                environment=env_vars,
            ),
            execution_role_arn=role_arn,
        )
        if model_cfg.get("network_isolation"):
            create_kwargs["enable_network_isolation"] = True
        model = Model.create(**create_kwargs)

        LOGGER.info(f"Creating endpoint config: {endpoint_name}")
        endpoint_config = EndpointConfig.create(
            endpoint_config_name=endpoint_name,
            production_variants=[
                ProductionVariant(
                    variant_name="AllTraffic",
                    model_name=endpoint_name,
                    initial_instance_count=1,
                    instance_pools=build_instance_pools(instance_types),
                    # Cap for walking the whole pool ladder on capacity errors.
                    variant_instance_provision_timeout_in_seconds=1800,
                    inference_ami_version=INFERENCE_AMI_VERSION,
                ),
            ],
        )

        LOGGER.info(f"Deploying endpoint: {endpoint_name} on {instance_types}")
        endpoint = Endpoint.create(
            endpoint_name=endpoint_name,
            endpoint_config_name=endpoint_name,
        )
        # Covers pool provisioning (<=1800s) plus model download and container boot.
        endpoint.wait_for_status("InService", timeout=4500)
    except Exception:
        _cleanup([endpoint, endpoint_config, model])
        raise

    LOGGER.info(f"Endpoint InService: {endpoint_name} on {instance_types}")
    return endpoint_name, model, endpoint_config, endpoint


def _generate_test_params():
    model_name_filter = os.environ.get("SM_MODEL_NAME")
    models = [m for m in load_sagemaker_config(model_name_filter) if "fleet" not in m]
    for m in models:
        # instance_type may be a single type or a priority-ordered ladder.
        m["instance_types"] = normalize_instance_types(m["instance_type"])
    return [(m["name"], m) for m in models]


@pytest.fixture(scope="module", params=_generate_test_params(), ids=lambda x: x[0])
def deployed_model(request, image_uri):
    """Deploy the model with an instance-pool ladder; SageMaker handles fallback.

    Insufficient capacity (ICE) on a single instance type is an AWS-side shortage, not
    an image defect. The configured ``instance_types`` become a production-variant
    instance pool, so SageMaker provisions the highest-priority type with capacity and
    falls back to the next server-side, absorbing the common case of one size being
    momentarily dry.

    If the endpoint never reaches InService the test fails. Skipping would leave the
    model unvalidated while the run still reported green, hiding a real coverage gap.
    """
    _, model_cfg = request.param

    region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")

    endpoint_name, model, endpoint_config, endpoint = _deploy_endpoint(
        image_uri, model_cfg, region, model_cfg["instance_types"]
    )
    try:
        yield {
            "endpoint_name": endpoint_name,
            "model_cfg": model_cfg,
            "region": region,
        }
    finally:
        _cleanup([endpoint, endpoint_config, model])


def test_model_serving(deployed_model):
    endpoint_name = deployed_model["endpoint_name"]
    model_cfg = deployed_model["model_cfg"]
    region = deployed_model["region"]

    sm_runtime = boto3.client("sagemaker-runtime", region_name=region)
    s3_client = boto3.client("s3", region_name=region)

    test_cases = model_cfg["test_cases"]
    failed = []

    for tc in test_cases:
        tc_name = tc["name"]
        LOGGER.info(f"--- Running: {tc_name} ---")

        content_type = tc.get("content_type", "application/json")
        request_cfg = tc["request"]

        if content_type.startswith("multipart/form-data"):
            body, ct = build_multipart_body(request_cfg, s3_client)
        else:
            body, ct = build_json_body(request_cfg)

        invoke_kwargs = {
            "EndpointName": endpoint_name,
            "Body": body,
            "ContentType": ct,
        }
        if "route" in tc:
            invoke_kwargs["CustomAttributes"] = f"route={tc['route']}"

        t0 = time.time()
        try:
            response = sm_runtime.invoke_endpoint(**invoke_kwargs)
            elapsed = time.time() - t0
            raw_bytes = response["Body"].read()

            try:
                response_body = json.loads(raw_bytes)
            except (json.JSONDecodeError, UnicodeDecodeError):
                response_body = None

            LOGGER.info(f"  Response ({elapsed:.1f}s, {len(raw_bytes)} bytes)")
            if response_body:
                LOGGER.info(f"  Body: {json.dumps(response_body)[:300]}")

            validate_response(response_body, raw_bytes, tc["validate"])
            LOGGER.info(f"  PASS: {tc_name}")

        except Exception as e:
            elapsed = time.time() - t0
            LOGGER.error(f"  FAIL: {tc_name} ({elapsed:.1f}s) — {e}")
            failed.append((tc_name, str(e)))

    if failed:
        msg = "\n".join(f"  - {name}: {err}" for name, err in failed)
        pytest.fail(f"{len(failed)}/{len(test_cases)} test cases failed:\n{msg}")
