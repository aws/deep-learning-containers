"""OpenFold3 SageMaker async-inference integration tests.

One 4-GPU endpoint (ml.g6.12xlarge) serves all cases. The handler's GPU pool
leases one GPU per concurrent request, so "1 GPU" == 1 in-flight request and
"4 GPU" == 4 concurrent requests.

Smoke validation only: a successful prediction returns at least one non-empty
CIF structure. OpenFold3 diffusion is stochastic, so exact-output matching is
not meaningful.

Shared fixtures (image_uri, region, aws_session) come from the root
test/conftest.py.
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

SMALL_QUERY = "small.json"  # ubiquitin, 76 residues
LARGE_QUERY = "large.json"  # synthetic, 622 residues

# concurrency-4 should run in roughly the same wall-time as concurrency-1
# because the 4 requests fan out across the 4 GPUs. Allow generous headroom
# for scheduling/IO jitter; serialized execution would be ~4x.
CONCURRENCY_SPEEDUP_TOLERANCE = 2.0


@pytest.fixture(scope="module")
def async_endpoint(aws_session, image_uri):
    """Deploy one 4-GPU async endpoint for the module; clean up after."""
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


@pytest.fixture(scope="module")
def invoke_async(aws_session, async_endpoint):
    """Return a helper that submits N async requests concurrently (one GPU each)
    and waits for all. Returns a list of (elapsed_seconds, parsed_result).

    Query JSON is read from s3://dlc-cicd-models/openfold3/queries/<query>.json
    (uploaded out of band) and copied to a unique input key per invocation.
    """
    s3 = aws_session.session.client("s3")
    smr = aws_session.session.client("sagemaker-runtime")
    endpoint_name = async_endpoint.endpoint_name

    def _submit(query_file: str) -> str:
        """Copy the query to a unique input key and kick off an async invocation. Returns output S3 URI."""
        body = s3.get_object(Bucket=S3_BUCKET, Key=f"{S3_INPUT_PREFIX}/{query_file}")["Body"].read()
        input_key = f"{S3_INPUT_PREFIX}/run/{uuid.uuid4()}-{query_file}"
        s3.put_object(Bucket=S3_BUCKET, Key=input_key, Body=body, ContentType="application/json")
        resp = smr.invoke_endpoint_async(
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
                s3.head_object(Bucket=bucket, Key=key)
            except s3.exceptions.ClientError:
                time.sleep(15)
                continue
            elapsed = time.time() - start
            body = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
            return elapsed, body
        raise TimeoutError(f"Async output {output_location} did not appear within {timeout}s")

    def helper(query_files: list[str]) -> list[tuple[float, dict]]:
        start = time.time()
        outputs = [_submit(q) for q in query_files]
        return [_wait(out, start) for out in outputs]

    return helper


def _assert_success(result: dict):
    """A successful OpenFold3 prediction has status 'success' and a non-empty CIF."""
    assert result.get("status") == "success", f"prediction failed: {result.get('error', result)}"
    structures = result.get("structures", [])
    assert structures, "no structures returned"
    assert structures[0].get("content"), "first structure has empty CIF content"


def test_small_single(invoke_async):
    """Smoke: small protein, single request on one GPU returns a valid structure."""
    ((elapsed, result),) = invoke_async([SMALL_QUERY])
    _assert_success(result)
    LOGGER.info(f"small x1 completed in {elapsed:.0f}s")


def test_large_single(invoke_async):
    """Large protein, single request on one GPU returns a valid structure."""
    ((elapsed, result),) = invoke_async([LARGE_QUERY])
    _assert_success(result)
    LOGGER.info(f"large x1 completed in {elapsed:.0f}s")


def test_large_concurrent_uses_gpu_pool(invoke_async):
    """Large protein, 1 request then 4 concurrent: the GPU pool should keep the
    4-concurrent wall-time under ~2x the single-request time (not ~4x)."""
    ((t1, result1),) = invoke_async([LARGE_QUERY])
    _assert_success(result1)

    results4 = invoke_async([LARGE_QUERY] * 4)
    for _, result in results4:
        _assert_success(result)
    t4 = max(elapsed for elapsed, _ in results4)

    LOGGER.info(f"large: 1-request={t1:.0f}s, 4-concurrent(max)={t4:.0f}s, ratio={t4 / t1:.2f}")
    assert t4 < CONCURRENCY_SPEEDUP_TOLERANCE * t1, (
        f"4 concurrent requests took {t4:.0f}s vs {t1:.0f}s for 1 "
        f"(ratio {t4 / t1:.2f} >= {CONCURRENCY_SPEEDUP_TOLERANCE}); GPU pool may not be parallelizing"
    )
