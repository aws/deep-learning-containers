"""Pytest fixtures for TF 2.20 inference integration tests on SageMaker.

Uses SageMaker Python SDK v3 resource layer (Model.create -> EndpointConfig ->
Endpoint -> endpoint.invoke). Fixtures defer AWS calls to test-execution time
so `pytest --collect-only` works without credentials.
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
    arn = os.environ.get("SM_ROLE_ARN")
    if not arn:
        pytest.skip("SM_ROLE_ARN not set")
    return arn


@pytest.fixture(scope="session")
def inference_image_uri() -> str:
    """ECR URI for the TF 2.20 inference image under test. Skips if not set."""
    uri = os.environ.get("TEST_IMAGE_URI")
    if not uri:
        pytest.skip("TEST_IMAGE_URI not set")
    return uri


# SM instance-type map keyed on the image's device_type. GPU is ml.g6.4xlarge
# to match the CI account's provisioned hosting fleet (ml.g5.xlarge is not
# hosted here and fails CreateEndpoint with CannotStartContainerError).
_SM_INSTANCE_TYPE_BY_DEVICE = {
    "cpu": "ml.c5.xlarge",
    "gpu": "ml.g6.4xlarge",
}


@pytest.fixture(scope="session")
def sm_instance_type() -> str:
    """SM endpoint instance type from SM_DEVICE_TYPE (cpu|gpu)."""
    device = os.environ.get("SM_DEVICE_TYPE", "").lower()
    assert device in {"cpu", "gpu"}, f"SM_DEVICE_TYPE must be 'cpu' or 'gpu'; got {device!r}."
    return _SM_INSTANCE_TYPE_BY_DEVICE[device]


@pytest.fixture(scope="session")
def boto_session(aws_region: str):
    """boto3 session bound to the configured region."""
    import boto3

    return boto3.Session(region_name=aws_region)


@pytest.fixture(scope="session")
def sagemaker_session(boto_session):
    """SageMaker SDK v3 session (default_bucket / upload_data)."""
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
def deploy_endpoint(
    boto_session,
    sagemaker_session,
    sagemaker_role_arn,
    inference_image_uri,
    sm_instance_type,
    unique_name,
    cleanup_endpoint,
):
    """Deploy a SageMaker endpoint; returns (endpoint, endpoint_name, model_name).

    Cleanup is registered BEFORE any AWS create call so a mid-flight failure
    (Model.create, EndpointConfig.create, Endpoint.create, wait_for_status)
    still gets torn down — otherwise a wait_for_status raise would skip a
    call-site cleanup and leak billing.
    """

    def _deploy(
        *,
        model_data_url: str,
        mode: str = "SingleModel",
        container_env: dict | None = None,
        name_prefix: str = "tf220-inference",
    ):
        from sagemaker.core.resources import (
            ContainerDefinition,
            Endpoint,
            EndpointConfig,
            Model,
            ProductionVariant,
        )

        endpoint_name = unique_name(name_prefix)
        model_name = unique_name(f"{name_prefix}-model")

        # Register cleanup BEFORE any AWS mutation.
        cleanup_endpoint(endpoint_name, model_name=model_name)

        container_kwargs = {
            "image": inference_image_uri,
            "model_data_url": model_data_url,
        }
        if mode == "MultiModel":
            container_kwargs["mode"] = "MultiModel"
        if container_env:
            container_kwargs["environment"] = dict(container_env)

        Model.create(
            model_name=model_name,
            primary_container=ContainerDefinition(**container_kwargs),
            execution_role_arn=sagemaker_role_arn,
            session=boto_session,
        )

        EndpointConfig.create(
            endpoint_config_name=endpoint_name,
            production_variants=[
                ProductionVariant(
                    variant_name="AllTraffic",
                    model_name=model_name,
                    initial_instance_count=1,
                    instance_type=sm_instance_type,
                ),
            ],
            session=boto_session,
        )

        endpoint = Endpoint.create(
            endpoint_name=endpoint_name,
            endpoint_config_name=endpoint_name,
            session=boto_session,
        )
        endpoint.wait_for_status("InService")
        return endpoint, endpoint_name, model_name

    return _deploy


@pytest.fixture
def cleanup_endpoint(boto_session):
    """Yield-style fixture that tears down endpoint, endpoint config, and model."""
    registered: list[dict] = []

    def _register(endpoint_name: str, model_name: str | None = None) -> None:
        registered.append({"endpoint_name": endpoint_name, "model_name": model_name})

    yield _register

    # Import lazily so collection works without the SDK installed.
    from sagemaker.core.resources import Endpoint, EndpointConfig, Model

    for item in registered:
        endpoint_name = item["endpoint_name"]
        model_name = item["model_name"]

        # Endpoint config name == endpoint name in our deploy flow.
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
