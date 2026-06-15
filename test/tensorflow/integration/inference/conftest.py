"""Pytest fixtures for TF 2.20 inference integration tests on SageMaker.

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
def sagemaker_session(boto_session):
    """A SageMaker SDK session for high-level deploy/predict calls."""
    import sagemaker

    return sagemaker.Session(boto_session=boto_session)


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
def cleanup_endpoint(sagemaker_session):
    """Yield-style fixture that tears down endpoint, endpoint config, and model.

    Usage:
        def test_x(cleanup_endpoint, ...):
            cleanup_endpoint(endpoint_name, model_name=model_name)
            # ... deploy + predict ...
    """
    registered: list[dict] = []

    def _register(endpoint_name: str, model_name: str | None = None) -> None:
        registered.append({"endpoint_name": endpoint_name, "model_name": model_name})

    yield _register

    sm_client = sagemaker_session.boto_session.client("sagemaker")
    for item in registered:
        endpoint_name = item["endpoint_name"]
        model_name = item["model_name"]

        for delete_call, kwargs in (
            (sm_client.delete_endpoint, {"EndpointName": endpoint_name}),
            (sm_client.delete_endpoint_config, {"EndpointConfigName": endpoint_name}),
        ):
            try:
                delete_call(**kwargs)
            except sm_client.exceptions.ClientError:
                # Swallow NotFound / already-deleted; teardown should be best-effort.
                pass
            except Exception:
                pass

        if model_name:
            try:
                sm_client.delete_model(ModelName=model_name)
            except Exception:
                pass
