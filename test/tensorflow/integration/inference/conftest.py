"""Pytest fixtures for TF 2.20 inference integration tests on SageMaker.

These fixtures use boto3 directly (sagemaker, sagemaker-runtime, s3 clients)
rather than the SageMaker Python SDK. The PyPI ``sagemaker`` package v3.x
removed the legacy v2 surfaces this suite relied on (``sagemaker.Session``,
``sagemaker.tensorflow.serving.TensorFlowModel``, ``sagemaker.multidatamodel``),
and the v3 ``ModelBuilder`` flow is heavier than what these integration tests
need. Going boto3-only keeps the tests transparent and SDK-version-independent.

Fixtures intentionally defer all AWS calls until test-execution time so that
``pytest --collect-only`` works in environments without AWS credentials.
"""

from __future__ import annotations

import os
import time
from uuid import uuid4

import pytest


@pytest.fixture(scope="session")
def aws_region() -> str:
    """AWS region for SageMaker operations. Defaults to us-west-2."""
    return os.environ.get("AWS_REGION", "us-west-2")


@pytest.fixture(scope="session")
def sagemaker_role_arn() -> str:
    """SageMaker execution role ARN. Skips the test if not set."""
    arn = os.environ.get("SAGEMAKER_ROLE_ARN")
    if not arn:
        pytest.skip("SAGEMAKER_ROLE_ARN not set")
    return arn


@pytest.fixture(scope="session")
def inference_image_uri() -> str:
    """ECR URI for the TF 2.20 inference image under test. Skips if not set."""
    uri = os.environ.get("INFERENCE_IMAGE_URI")
    if not uri:
        pytest.skip("INFERENCE_IMAGE_URI not set")
    return uri


@pytest.fixture(scope="session")
def boto_session(aws_region: str):
    """A boto3 session bound to the configured region."""
    import boto3

    return boto3.Session(region_name=aws_region)


@pytest.fixture(scope="session")
def sagemaker_client(boto_session):
    """Low-level SageMaker control-plane client (create/delete model, endpoint, ...)."""
    return boto_session.client("sagemaker")


@pytest.fixture(scope="session")
def sagemaker_runtime_client(boto_session):
    """SageMaker runtime client used to invoke endpoints."""
    return boto_session.client("sagemaker-runtime")


@pytest.fixture(scope="session")
def s3_client(boto_session):
    """S3 client used to upload sample model tarballs."""
    return boto_session.client("s3")


@pytest.fixture(scope="session")
def default_bucket(boto_session, aws_region: str, s3_client) -> str:
    """Resolve the ``sagemaker-<region>-<account>`` default bucket, creating it if absent.

    Mirrors the behaviour of the v2 SDK's ``Session.default_bucket()`` so test
    bodies can keep using a single, predictable bucket without callers having
    to plumb one in.
    """
    sts = boto_session.client("sts")
    account_id = sts.get_caller_identity()["Account"]
    bucket = f"sagemaker-{aws_region}-{account_id}"

    try:
        s3_client.head_bucket(Bucket=bucket)
    except Exception:
        # Bucket missing or inaccessible — try to create it. us-east-1 must omit
        # LocationConstraint; every other region requires it.
        create_kwargs: dict = {"Bucket": bucket}
        if aws_region != "us-east-1":
            create_kwargs["CreateBucketConfiguration"] = {"LocationConstraint": aws_region}
        try:
            s3_client.create_bucket(**create_kwargs)
        except Exception:
            # Race or pre-existing-but-403 — leave the original error to surface
            # at upload time rather than masking it here.
            pass

    return bucket


@pytest.fixture
def unique_name():
    """Returns a callable producing collision-resistant resource names.

    Usage:
        name = unique_name("tf220-single")
    """

    def _make(prefix: str) -> str:
        return f"{prefix}-{int(time.time())}-{uuid4().hex[:6]}"

    return _make


@pytest.fixture
def upload_to_s3(s3_client):
    """Yield-style helper that uploads a local file to ``s3://<bucket>/<key>``.

    Returns the resulting ``s3://`` URI. Failures propagate; teardown is
    intentionally not provided since SageMaker model artifacts are typically
    left in the bucket for forensic inspection.
    """

    def _upload(local_path: str, bucket: str, key: str) -> str:
        s3_client.upload_file(local_path, bucket, key)
        return f"s3://{bucket}/{key}"

    return _upload


@pytest.fixture
def wait_for_endpoint(sagemaker_client):
    """Poll ``describe_endpoint`` until the endpoint reaches ``InService``.

    Raises ``RuntimeError`` if the endpoint enters ``Failed`` / ``OutOfService``
    or if the wait exceeds ``timeout_seconds`` (default 1800s = 30 min, which
    matches typical first-pull cold-start latency for inference DLCs).
    """

    def _wait(endpoint_name: str, timeout_seconds: int = 1800, poll_seconds: int = 30) -> None:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            resp = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            status = resp["EndpointStatus"]
            if status == "InService":
                return
            if status in {"Failed", "OutOfService"}:
                reason = resp.get("FailureReason", "<no FailureReason>")
                raise RuntimeError(
                    f"endpoint {endpoint_name} entered terminal state {status}: {reason}"
                )
            time.sleep(poll_seconds)
        raise RuntimeError(
            f"endpoint {endpoint_name} did not reach InService within {timeout_seconds}s"
        )

    return _wait


@pytest.fixture
def cleanup_endpoint(sagemaker_client):
    """Yield-style fixture that tears down endpoint, endpoint config, and model.

    Usage:
        def test_x(cleanup_endpoint, ...):
            cleanup_endpoint(endpoint_name, model_name=model_name)
            # ... deploy + predict ...

    Endpoint config is registered under the same name as the endpoint, matching
    what the test bodies pass to ``create_endpoint_config``.
    """
    registered: list[dict] = []

    def _register(endpoint_name: str, model_name: str | None = None) -> None:
        registered.append({"endpoint_name": endpoint_name, "model_name": model_name})

    yield _register

    for item in registered:
        endpoint_name = item["endpoint_name"]
        model_name = item["model_name"]

        for delete_call, kwargs in (
            (sagemaker_client.delete_endpoint, {"EndpointName": endpoint_name}),
            (sagemaker_client.delete_endpoint_config, {"EndpointConfigName": endpoint_name}),
        ):
            try:
                delete_call(**kwargs)
            except Exception:
                # Swallow NotFound / already-deleted; teardown is best-effort.
                pass

        if model_name:
            try:
                sagemaker_client.delete_model(ModelName=model_name)
            except Exception:
                pass
