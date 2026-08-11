"""Pytest fixtures for TF 2.20 inference integration tests on SageMaker.

Uses SageMaker Python SDK v3 resource layer (Model.create -> EndpointConfig ->
Endpoint -> endpoint.invoke). Relies on parent test/conftest.py for image_uri,
region, and aws_session fixtures (passed via --image-uri / --region CLI args).
"""

from __future__ import annotations

import logging
import os

import pytest
from test_utils import random_suffix_name
from test_utils.constants import SAGEMAKER_ROLE

LOGGER = logging.getLogger(__name__)

# SM instance-type map keyed on the image's device_type. GPU is ml.g6.4xlarge
# to match the CI account's provisioned hosting fleet.
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
def boto_session(region):
    """boto3 session bound to the configured region."""
    import boto3

    return boto3.Session(region_name=region)


@pytest.fixture(scope="session")
def sagemaker_session(boto_session):
    """SageMaker SDK v3 session (default_bucket / upload_data)."""
    from sagemaker.core.helper.session_helper import Session

    return Session(boto_session=boto_session)


@pytest.fixture(scope="session")
def sagemaker_role_arn(aws_session) -> str:
    """Resolve SAGEMAKER_ROLE constant to full ARN via AWSSessionManager."""
    return aws_session.resolve_role_arn(SAGEMAKER_ROLE)


def _cleanup(resources, boto_session):
    """Best-effort delete for a list of (resource_cls, get_kwargs) tuples (None-safe)."""
    for resource_cls, get_kwargs in resources:
        if any(v is None for v in get_kwargs.values()):
            continue
        try:
            resource_cls.get(session=boto_session, **get_kwargs).delete()
        except Exception as e:
            LOGGER.warning(f"Cleanup {resource_cls.__name__} failed: {e}")


@pytest.fixture
def deploy_endpoint(
    boto_session,
    sagemaker_session,
    sagemaker_role_arn,
    image_uri,
    sm_instance_type,
):
    """Deploy a SageMaker endpoint; yields (endpoint, endpoint_name, model_name).

    Uses try/finally so partially-created resources are always torn down —
    even if deployment fails mid-flight (prevents billing leaks).
    """
    from sagemaker.core.resources import (
        ContainerDefinition,
        Endpoint,
        EndpointConfig,
        Model,
        ProductionVariant,
    )

    model = endpoint_config = endpoint = None
    endpoint_name = model_name = None

    def _deploy(
        *,
        model_data_url: str,
        mode: str = "SingleModel",
        container_env: dict | None = None,
        name_prefix: str = "tf220-inference",
    ):
        nonlocal model, endpoint_config, endpoint, endpoint_name, model_name

        endpoint_name = random_suffix_name(name_prefix, 63)
        model_name = random_suffix_name(f"{name_prefix}-model", 63)

        container_kwargs = {
            "image": image_uri,
            "model_data_url": model_data_url,
        }
        if mode == "MultiModel":
            container_kwargs["mode"] = "MultiModel"
        if container_env:
            container_kwargs["environment"] = dict(container_env)

        model = Model.create(
            model_name=model_name,
            primary_container=ContainerDefinition(**container_kwargs),
            execution_role_arn=sagemaker_role_arn,
            session=boto_session,
        )

        endpoint_config = EndpointConfig.create(
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

    try:
        yield _deploy
    finally:
        _cleanup(
            [
                (Endpoint, {"endpoint_name": endpoint_name}),
                (EndpointConfig, {"endpoint_config_name": endpoint_name}),
                (Model, {"model_name": model_name}),
            ],
            boto_session,
        )
