"""OpenFold3 SageMaker async-inference integration tests (SDK v3, no local GPU).

Env vars (set by the workflow): TEST_IMAGE_URI, SM_ROLE_ARN, AWS_REGION.
One 4-GPU endpoint serves all cases; a concurrent request leases one GPU each.
Query inputs and async output use the account's sagemaker-<region>-<account>
bucket (the only S3 the SageMaker role can read/write). Smoke validation only:
a successful prediction returns a non-empty CIF (diffusion is non-deterministic).
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

# One 4-GPU instance hosts every case; 1-GPU cases just send 1 request.
INSTANCE_TYPE = "ml.g6.12xlarge"
MAX_CONCURRENT_INVOCATIONS = 4
# Generous: warmup compiles CUDA kernels (~6 min) before /ping returns 200.
STARTUP_HEALTH_CHECK_TIMEOUT = 1200

# Bucket must be writable by the SageMaker role -> account default sagemaker bucket.
_account = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]
BUCKET = os.environ.get("SM_IO_BUCKET", f"sagemaker-{REGION}-{_account}")
QUERY_PREFIX = "openfold3/queries"
OUTPUT_PREFIX = "openfold3/async-output"

SMALL_QUERY = "small.json"  # ubiquitin, 76 residues
LARGE_QUERY = "large.json"  # synthetic, 622 residues

# Bounded so a stuck request fails fast instead of hanging.
POLL_TIMEOUT = 900
POLL_INTERVAL = 5
# Defensive only; not a documented SageMaker requirement.
SETTLE_SECONDS = 30

# 4-concurrent should be ~1x the single-request time, not ~4x.
CONCURRENCY_SPEEDUP_TOLERANCE = 2.0

_s3 = boto3.client("s3", region_name=REGION)
_smr = boto3.client("sagemaker-runtime", region_name=REGION)


def _preflight_s3():
    """Fail before the ~10-min deploy if queries are missing or the bucket is unwritable."""
    for q in (SMALL_QUERY, LARGE_QUERY):
        try:
            _s3.head_object(Bucket=BUCKET, Key=f"{QUERY_PREFIX}/{q}")
        except _s3.exceptions.ClientError as e:
            raise AssertionError(
                f"Query input s3://{BUCKET}/{QUERY_PREFIX}/{q} not found ({e}). "
                f"Upload the query JSONs to the sagemaker bucket first."
            ) from e
    probe = f"{OUTPUT_PREFIX}/.preflight-{uuid.uuid4()}"
    try:
        _s3.put_object(Bucket=BUCKET, Key=probe, Body=b"ok")
        _s3.delete_object(Bucket=BUCKET, Key=probe)
    except _s3.exceptions.ClientError as e:
        raise AssertionError(
            f"Cannot write to s3://{BUCKET}/{OUTPUT_PREFIX}/ ({e}). The async "
            f"endpoint output will not be writable — pick a bucket the SageMaker "
            f"execution role can write (name must contain 'sagemaker')."
        ) from e


def _submit(endpoint_name: str, query_file: str) -> tuple[str, str]:
    """Invoke the async endpoint for a query already in the bucket. Returns (output_uri, failure_uri)."""
    resp = _smr.invoke_endpoint_async(
        EndpointName=endpoint_name,
        InputLocation=f"s3://{BUCKET}/{QUERY_PREFIX}/{query_file}",
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


def _invoke(endpoint_name: str, query_files: list[str]) -> list[tuple[float, dict]]:
    """Submit N requests concurrently, wait for all; returns [(elapsed_s, result)]."""
    start = time.time()
    submitted = [_submit(endpoint_name, q) for q in query_files]
    return [_poll_one(out, fail, start) for out, fail in submitted]


@pytest.fixture(scope="module")
def endpoint_name():
    """Deploy one 4-GPU async endpoint for the module; sweep stale first, clean up after."""
    _preflight_s3()
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
                    s3_output_path=f"s3://{BUCKET}/{OUTPUT_PREFIX}/",
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
    ((elapsed, result),) = _invoke(endpoint_name, [SMALL_QUERY])
    _assert_success(result)
    LOGGER.info(f"small x1 completed in {elapsed:.0f}s")


def test_large_single(endpoint_name):
    """Large protein, single request on one GPU returns a valid structure."""
    ((elapsed, result),) = _invoke(endpoint_name, [LARGE_QUERY])
    _assert_success(result)
    LOGGER.info(f"large x1 completed in {elapsed:.0f}s")


def test_large_concurrent_uses_gpu_pool(endpoint_name):
    """Large protein: 4-concurrent wall-time stays under ~2x the single-request time."""
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
