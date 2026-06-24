"""Fixtures for the OpenFold3 SageMaker async-inference test.

Deploys ONE 4-GPU async endpoint (ml.g6.12xlarge) once per session and reuses it
across all test cases. Model weights are baked into the image, so no ModelDataUrl
is set; async inference only needs an S3 path for request/response payloads.
"""

import json
import logging
import time
import uuid

import pytest
from sagemaker.core.resources import Endpoint, EndpointConfig, Model
from sagemaker.core.shapes import (
    AsyncInferenceClientConfig,
    AsyncInferenceConfig,
    AsyncInferenceOutputConfig,
    ContainerDefinition,
    ProductionVariant,
)
from test_utils import random_suffix_name
from test_utils.constants import INFERENCE_AMI_VERSION, SAGEMAKER_ROLE

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# Single 4-GPU instance hosts every case (1-GPU cases just send 1 request).
INSTANCE_TYPE = "ml.g6.12xlarge"
# Allow up to 4 concurrent invocations so the handler's GPU pool can fan out
# across all 4 GPUs (the POC default of 1 would serialize requests).
MAX_CONCURRENT_INVOCATIONS = 4
# Warmup compiles CUDA kernels (~6 min) before /ping returns 200; give the
# container startup health check generous headroom.
STARTUP_HEALTH_CHECK_TIMEOUT = 1200

S3_BUCKET = "dlc-cicd-models"
S3_INPUT_PREFIX = "openfold3/queries"
S3_OUTPUT_PREFIX = "openfold3/async-output"


@pytest.fixture(scope="session")
def s3_client(aws_session):
    return aws_session.session.client("s3")


@pytest.fixture(scope="session")
def sagemaker_runtime(aws_session):
    return aws_session.session.client("sagemaker-runtime")


@pytest.fixture(scope="session")
def async_endpoint(aws_session, image_uri, region):
    """Deploy one 4-GPU async endpoint for the whole session; clean up after."""
    role_arn = aws_session.resolve_role_arn(SAGEMAKER_ROLE)
    name = random_suffix_name("openfold3-async", 50)

    model = endpoint_config = endpoint = None
    try:
        LOGGER.info(f"Creating model: {name}")
        model = Model.create(
            model_name=name,
            primary_container=ContainerDefinition(
                image=image_uri,
                environment={"OPENFOLD_CACHE": "/root/.openfold3"},
            ),
            execution_role_arn=role_arn,
        )

        LOGGER.info(f"Creating async endpoint config: {name}")
        endpoint_config = EndpointConfig.create(
            endpoint_config_name=name,
            production_variants=[
                ProductionVariant(
                    variant_name="AllTraffic",
                    model_name=name,
                    initial_instance_count=1,
                    instance_type=INSTANCE_TYPE,
                    inference_ami_version=INFERENCE_AMI_VERSION,
                    container_startup_health_check_timeout_in_seconds=STARTUP_HEALTH_CHECK_TIMEOUT,
                ),
            ],
            async_inference_config=AsyncInferenceConfig(
                output_config=AsyncInferenceOutputConfig(
                    s3_output_path=f"s3://{S3_BUCKET}/{S3_OUTPUT_PREFIX}/",
                ),
                client_config=AsyncInferenceClientConfig(
                    max_concurrent_invocations_per_instance=MAX_CONCURRENT_INVOCATIONS,
                ),
            ),
        )

        LOGGER.info(f"Deploying endpoint {name} (~10-15 min incl. warmup)...")
        endpoint = Endpoint.create(endpoint_name=name, endpoint_config_name=name)
        endpoint.wait_for_status("InService")
        LOGGER.info("Endpoint InService")

        yield endpoint
    finally:
        for resource in (endpoint, endpoint_config, model):
            if resource is None:
                continue
            try:
                resource.delete()
            except Exception as e:
                LOGGER.warning(f"Cleanup {type(resource).__name__} failed: {e}")


@pytest.fixture(scope="session")
def invoke_async(aws_session, s3_client, sagemaker_runtime, async_endpoint):
    """Return a helper that submits one async request and returns elapsed seconds + parsed result.

    The query JSON is read from s3://dlc-cicd-models/openfold3/queries/<query>.json
    (uploaded out of band) and copied to a unique input key per invocation.
    """
    endpoint_name = async_endpoint.endpoint_name

    def _submit(query_file: str) -> str:
        """Copy the query to a unique input key and kick off an async invocation. Returns output S3 URI."""
        src_key = f"{S3_INPUT_PREFIX}/{query_file}"
        body = s3_client.get_object(Bucket=S3_BUCKET, Key=src_key)["Body"].read()
        input_key = f"{S3_INPUT_PREFIX}/run/{uuid.uuid4()}-{query_file}"
        s3_client.put_object(
            Bucket=S3_BUCKET, Key=input_key, Body=body, ContentType="application/json"
        )
        resp = sagemaker_runtime.invoke_endpoint_async(
            EndpointName=endpoint_name,
            InputLocation=f"s3://{S3_BUCKET}/{input_key}",
            ContentType="application/json",
        )
        return resp["OutputLocation"]

    def _wait(output_location: str, start: float, timeout: int = 3600) -> tuple[float, dict]:
        """Poll S3 until the async output appears. Returns (elapsed_seconds, parsed_body)."""
        bucket = output_location.split("/")[2]
        key = "/".join(output_location.split("/")[3:])
        deadline = start + timeout
        while time.time() < deadline:
            try:
                s3_client.head_object(Bucket=bucket, Key=key)
            except s3_client.exceptions.ClientError:
                time.sleep(15)
                continue
            elapsed = time.time() - start
            body = json.loads(s3_client.get_object(Bucket=bucket, Key=key)["Body"].read())
            return elapsed, body
        raise TimeoutError(f"Async output {output_location} did not appear within {timeout}s")

    def helper(query_files: list[str]) -> list[tuple[float, dict]]:
        """Submit N requests concurrently (one GPU each), wait for all. Returns list of (elapsed, result)."""
        start = time.time()
        outputs = [_submit(q) for q in query_files]
        return [_wait(out, start) for out in outputs]

    return helper
