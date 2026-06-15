"""Pytest fixtures for TF 2.20 inference integration tests on SageMaker.

Uses the SageMaker Python SDK v3 (``sagemaker>=3.0.0``) — the v2 Estimator /
Model / Predictor classes were removed in v3 in favor of the unified
``ModelBuilder`` and the ``sagemaker-core`` resource layer
(``Endpoint``, ``EndpointConfig``, ``Model``, ``ContainerDefinition``,
``ProductionVariant``). For these DLC tests we already have a custom
``image_uri`` and a pre-built ``model.tar.gz``, so the simplest v3 path is
the resource layer directly: ``Model.create -> EndpointConfig.create ->
Endpoint.create -> endpoint.invoke()``. ``ModelBuilder`` is the right choice
when the SDK should auto-detect the framework / container / packaging — for
us, those are all fixed by the test fixture inputs.

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
    """A boto3 session bound to the configured region.

    Used purely as a transport for ``sagemaker.core.helper.session_helper.Session``
    and for the underlying ``s3`` client when uploading model artifacts; no
    SageMaker control-plane calls go through it directly.
    """
    import boto3

    return boto3.Session(region_name=aws_region)


@pytest.fixture(scope="session")
def sagemaker_session(boto_session):
    """A SageMaker SDK v3 session.

    ``sagemaker.core.helper.session_helper.Session`` is the v3 replacement for
    the v2 ``sagemaker.Session``. We use it for ``default_bucket()`` and
    ``upload_data()``; resource-layer ``create()`` calls accept it via the
    ``session=`` kwarg.
    """
    from sagemaker.core.helper.session_helper import Session

    return Session(boto_session=boto_session)


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
def cleanup_endpoint(boto_session):
    """Yield-style fixture that tears down endpoint, endpoint config, and model.

    Uses the v3 ``sagemaker-core`` resource layer (``Endpoint.get(...).delete()``,
    etc.) rather than raw boto3 SDK calls, so cleanup code matches the deploy
    code in the tests. The ``session=`` kwarg on resource ``get`` / ``create``
    methods accepts a raw ``boto3.session.Session`` (see
    ``sagemaker.core.utils.utils.SageMakerClient``); pass ``boto_session``
    rather than the helper ``Session``.

    Usage:
        def test_x(cleanup_endpoint, ...):
            cleanup_endpoint(endpoint_name, model_name=model_name)
            # ... deploy + predict ...
    """
    registered: list[dict] = []

    def _register(endpoint_name: str, model_name: str | None = None) -> None:
        registered.append({"endpoint_name": endpoint_name, "model_name": model_name})

    yield _register

    # Import lazily so collection works without the SDK installed.
    from sagemaker.core.resources import Endpoint, EndpointConfig, Model

    for item in registered:
        endpoint_name = item["endpoint_name"]
        model_name = item["model_name"]

        # Endpoint config name == endpoint name in our deploy flow below.
        for resource_cls, get_kwargs in (
            (Endpoint, {"endpoint_name": endpoint_name}),
            (EndpointConfig, {"endpoint_config_name": endpoint_name}),
        ):
            try:
                resource_cls.get(session=boto_session, **get_kwargs).delete()
            except Exception:
                # Best-effort teardown: swallow NotFound / already-deleted.
                pass

        if model_name:
            try:
                Model.get(model_name=model_name, session=boto_session).delete()
            except Exception:
                pass
