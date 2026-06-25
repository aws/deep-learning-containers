"""OpenFold3 SageMaker async-inference integration tests.

Uses SageMaker Python SDK v3. Launches a real async endpoint — no GPU needed on
the runner. Configured via env vars (set by the workflow):
  TEST_IMAGE_URI  image to deploy
  SM_ROLE_ARN     SageMaker execution role ARN
  AWS_REGION      region (default us-west-2)

One 4-GPU endpoint (ml.g6.12xlarge) serves all cases. The handler's GPU pool
leases one GPU per concurrent request, so "1 GPU" == 1 in-flight request and
"4 GPU" == 4 concurrent requests.

Async I/O bucket: the SageMaker execution role (AmazonSageMakerFullAccess) can
only write to buckets whose name contains "sagemaker", so both the async input
and output objects go to the account's default sagemaker-<region>-<account>
bucket. The committed query JSONs are staged there at test time.

Smoke validation only: a successful prediction returns at least one non-empty
CIF structure. OpenFold3 diffusion is stochastic, so exact-output matching is
not meaningful.
"""

import json
import logging
import os
import time
import uuid

import boto3
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
from test_utils.constants import INFERENCE_AMI_VERSION

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

IMAGE_URI = os.environ["TEST_IMAGE_URI"]
ROLE_ARN = os.environ["SM_ROLE_ARN"]
REGION = os.environ.get("AWS_REGION", "us-west-2")

# Single 4-GPU instance hosts every case (1-GPU cases just send 1 request).
INSTANCE_TYPE = "ml.g6.12xlarge"
# Allow up to 4 concurrent invocations so the handler's GPU pool can fan out
# across all 4 GPUs (the POC default of 1 would serialize requests).
MAX_CONCURRENT_INVOCATIONS = 4
# Warmup compiles CUDA kernels (~6 min) before /ping returns 200; give the
# container startup health check generous headroom.
STARTUP_HEALTH_CHECK_TIMEOUT = 1200

# Source of the committed query inputs (read-only is enough; the SageMaker role
# can GetObject here). They are re-staged into the sagemaker I/O bucket below.
QUERY_SRC_BUCKET = os.environ.get("QUERY_SRC_BUCKET", "dlc-cicd-models")
QUERY_SRC_PREFIX = os.environ.get("QUERY_SRC_PREFIX", "openfold3/queries")
# Async input/output bucket — must be writable by the SageMaker role, i.e. the
# account's default "sagemaker-<region>-<account>" bucket.
_account = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]
IO_BUCKET = f"sagemaker-{REGION}-{_account}"
IO_INPUT_PREFIX = "openfold3/async-input"
IO_OUTPUT_PREFIX = "openfold3/async-output"

SMALL_QUERY = "small.json"  # ubiquitin, 76 residues
LARGE_QUERY = "large.json"  # synthetic, 622 residues

# Per-request poll: an output should appear in a few minutes for warm requests.
# 900s is generous but bounded so a dropped request fails fast (not a 1h hang).
POLL_TIMEOUT = 900
POLL_INTERVAL = 5
# SageMaker async can drop the very first request submitted immediately after
# InService (the async queue poller may not be fully registered yet). Retry.
FIRST_REQUEST_RETRIES = 2
# Let the endpoint settle after InService before the first invocation.
SETTLE_SECONDS = 30

# concurrency-4 should run in roughly the same wall-time as concurrency-1
# because the 4 requests fan out across the 4 GPUs. Allow generous headroom
# for scheduling/IO jitter; serialized execution would be ~4x.
CONCURRENCY_SPEEDUP_TOLERANCE = 2.0

_s3 = boto3.client("s3", region_name=REGION)
_smr = boto3.client("sagemaker-runtime", region_name=REGION)


def _ensure_io_bucket():
    """Ensure the sagemaker I/O bucket exists (it normally does in any SM account)."""
    try:
        _s3.head_bucket(Bucket=IO_BUCKET)
    except _s3.exceptions.ClientError:
        kwargs = {"Bucket": IO_BUCKET}
        if REGION != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {"LocationConstraint": REGION}
        _s3.create_bucket(**kwargs)


def _stage_query(query_file: str) -> str:
    """Copy a committed query into the I/O bucket as a unique input object. Returns its key."""
    body = _s3.get_object(Bucket=QUERY_SRC_BUCKET, Key=f"{QUERY_SRC_PREFIX}/{query_file}")[
        "Body"
    ].read()
    key = f"{IO_INPUT_PREFIX}/{uuid.uuid4()}-{query_file}"
    _s3.put_object(Bucket=IO_BUCKET, Key=key, Body=body, ContentType="application/json")
    return key


def _submit(endpoint_name: str, input_key: str) -> tuple[str, str]:
    """Invoke the async endpoint for an already-staged input. Returns (output_uri, failure_uri)."""
    resp = _smr.invoke_endpoint_async(
        EndpointName=endpoint_name,
        InputLocation=f"s3://{IO_BUCKET}/{input_key}",
        ContentType="application/json",
    )
    return resp["OutputLocation"], resp.get("FailureLocation", "")


def _split(uri: str) -> tuple[str, str]:
    return uri.split("/")[2], "/".join(uri.split("/")[3:])


def _poll_one(output_uri: str, failure_uri: str, start: float) -> tuple[float, dict]:
    """Poll for the output object; raise if SageMaker writes a failure object or we time out."""
    ob, ok = _split(output_uri)
    fb, fk = _split(failure_uri) if failure_uri else (None, None)
    deadline = start + POLL_TIMEOUT
    while time.time() < deadline:
        try:
            _s3.head_object(Bucket=ob, Key=ok)
            elapsed = time.time() - start
            body = json.loads(_s3.get_object(Bucket=ob, Key=ok)["Body"].read())
            return elapsed, body
        except _s3.exceptions.ClientError:
            pass
        if fb:
            try:
                _s3.head_object(Bucket=fb, Key=fk)
                err = _s3.get_object(Bucket=fb, Key=fk)["Body"].read().decode()[:2000]
                raise AssertionError(f"SageMaker async failure object: {err}")
            except _s3.exceptions.ClientError:
                pass
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"No async output at {output_uri} within {POLL_TIMEOUT}s")


def _invoke(
    endpoint_name: str, query_files: list[str], retries: int = 0
) -> list[tuple[float, dict]]:
    """Submit N requests concurrently (one GPU each) and wait for all.

    If `retries` > 0, a TimeoutError on the batch (a silently-dropped request,
    common for the first invocation right after InService) triggers a resubmit.
    Returns a list of (elapsed_seconds, parsed_result).
    """
    attempt = 0
    while True:
        start = time.time()
        staged = [_stage_query(q) for q in query_files]
        submitted = [_submit(endpoint_name, key) for key in staged]
        try:
            return [_poll_one(out, fail, start) for out, fail in submitted]
        except TimeoutError:
            if attempt >= retries:
                raise
            attempt += 1
            LOGGER.warning(f"Invocation timed out; retrying ({attempt}/{retries})")


@pytest.fixture(scope="module")
def endpoint_name():
    """Deploy one 4-GPU async endpoint for the module; clean up after.

    Cleanup-first: sweep any stale openfold3-async-* resources before creating,
    so a previously-canceled run never blocks or strands this one.
    """
    _ensure_io_bucket()
    _sweep_stale()

    name = random_suffix_name("openfold3-async", 50)
    model = endpoint_config = endpoint = None
    try:
        LOGGER.info(f"Creating model: {name}")
        model = Model.create(
            model_name=name,
            primary_container=ContainerDefinition(
                image=IMAGE_URI,
                environment={"OPENFOLD_CACHE": "/root/.openfold3"},
            ),
            execution_role_arn=ROLE_ARN,
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
                    s3_output_path=f"s3://{IO_BUCKET}/{IO_OUTPUT_PREFIX}/",
                ),
                client_config=AsyncInferenceClientConfig(
                    max_concurrent_invocations_per_instance=MAX_CONCURRENT_INVOCATIONS,
                ),
            ),
        )

        LOGGER.info(f"Deploying endpoint {name} (~10-15 min incl. warmup)...")
        endpoint = Endpoint.create(endpoint_name=name, endpoint_config_name=name)
        endpoint.wait_for_status("InService")
        LOGGER.info(f"Endpoint InService; settling {SETTLE_SECONDS}s before first invocation")
        time.sleep(SETTLE_SECONDS)

        yield name
    finally:
        for resource in (endpoint, endpoint_config, model):
            if resource is None:
                continue
            try:
                resource.delete()
            except Exception as e:
                LOGGER.warning(f"Cleanup {type(resource).__name__} failed: {e}")


def _sweep_stale():
    """Delete any leftover openfold3-async-* SageMaker resources from a prior canceled run."""
    sm = boto3.client("sagemaker", region_name=REGION)
    try:
        for ep in sm.list_endpoints(NameContains="openfold3-async").get("Endpoints", []):
            LOGGER.warning(f"[sweep] deleting stale endpoint {ep['EndpointName']}")
            try:
                sm.delete_endpoint(EndpointName=ep["EndpointName"])
            except Exception as e:
                LOGGER.warning(f"[sweep] {e}")
        for c in sm.list_endpoint_configs(NameContains="openfold3-async").get(
            "EndpointConfigs", []
        ):
            try:
                sm.delete_endpoint_config(EndpointConfigName=c["EndpointConfigName"])
            except Exception as e:
                LOGGER.warning(f"[sweep] {e}")
        for m in sm.list_models(NameContains="openfold3-async").get("Models", []):
            try:
                sm.delete_model(ModelName=m["ModelName"])
            except Exception as e:
                LOGGER.warning(f"[sweep] {e}")
    except Exception as e:
        LOGGER.warning(f"[sweep] skipped: {e}")


def _assert_success(result: dict):
    """A successful OpenFold3 prediction has status 'success' and a non-empty CIF."""
    assert result.get("status") == "success", f"prediction failed: {result.get('error', result)}"
    structures = result.get("structures", [])
    assert structures, "no structures returned"
    assert structures[0].get("content"), "first structure has empty CIF content"


def test_small_single(endpoint_name):
    """Smoke: small protein, single request on one GPU returns a valid structure."""
    ((elapsed, result),) = _invoke(endpoint_name, [SMALL_QUERY], retries=FIRST_REQUEST_RETRIES)
    _assert_success(result)
    LOGGER.info(f"small x1 completed in {elapsed:.0f}s")


def test_large_single(endpoint_name):
    """Large protein, single request on one GPU returns a valid structure."""
    ((elapsed, result),) = _invoke(endpoint_name, [LARGE_QUERY])
    _assert_success(result)
    LOGGER.info(f"large x1 completed in {elapsed:.0f}s")


def test_large_concurrent_uses_gpu_pool(endpoint_name):
    """Large protein, 1 request then 4 concurrent: the GPU pool should keep the
    4-concurrent wall-time under ~2x the single-request time (not ~4x)."""
    ((t1, result1),) = _invoke(endpoint_name, [LARGE_QUERY])
    _assert_success(result1)

    results4 = _invoke(endpoint_name, [LARGE_QUERY] * 4)
    for _, result in results4:
        _assert_success(result)
    t4 = max(elapsed for elapsed, _ in results4)

    LOGGER.info(f"large: 1-request={t1:.0f}s, 4-concurrent(max)={t4:.0f}s, ratio={t4 / t1:.2f}")
    assert t4 < CONCURRENCY_SPEEDUP_TOLERANCE * t1, (
        f"4 concurrent requests took {t4:.0f}s vs {t1:.0f}s for 1 "
        f"(ratio {t4 / t1:.2f} >= {CONCURRENCY_SPEEDUP_TOLERANCE}); GPU pool may not be parallelizing"
    )
