"""Pytest fixtures for TF 2.20 inference integration tests on SageMaker.

Uses SageMaker Python SDK v3 resource layer (Model.create -> EndpointConfig ->
Endpoint -> endpoint.invoke). Relies on parent test/conftest.py for image_uri,
region, aws_session, and sagemaker_session fixtures (passed via --image-uri /
--region CLI args).
"""

from __future__ import annotations

import logging
import os

import pytest
from test_utils import random_suffix_name
from test_utils.constants import INFERENCE_AMI_VERSION_CU12, SAGEMAKER_ROLE

LOGGER = logging.getLogger(__name__)

# SM instance-type map keyed on the image's device_type. GPU is ml.g6.4xlarge
# to match the CI account's provisioned hosting fleet.
_SM_INSTANCE_TYPE_BY_DEVICE = {
    "cpu": "ml.c5.xlarge",
    "gpu": "ml.g6.4xlarge",
}


@pytest.fixture(scope="session")
def sm_device_type() -> str:
    """Device type from SM_DEVICE_TYPE env var (cpu|gpu). Fail-closed on misconfig."""
    device = os.environ.get("SM_DEVICE_TYPE", "").lower()
    assert device in {"cpu", "gpu"}, f"SM_DEVICE_TYPE must be 'cpu' or 'gpu'; got {device!r}."
    return device


@pytest.fixture(scope="session")
def sm_instance_type(sm_device_type) -> str:
    """SM endpoint instance type derived from device type."""
    return _SM_INSTANCE_TYPE_BY_DEVICE[sm_device_type]


def _cleanup(resources):
    """Best-effort delete for a list of v3 resource objects (None-safe)."""
    for resource in resources:
        if resource is None:
            continue
        try:
            resource.delete()
        except Exception as e:
            LOGGER.warning(f"Cleanup {type(resource).__name__} failed: {e}")


def _provision_endpoint(
    *,
    resources: list,
    session,
    role_arn: str,
    image_uri: str,
    sm_instance_type: str,
    sm_device_type: str,
    model_data_url: str,
    mode: str = "SingleModel",
    container_env: dict | None = None,
    name_prefix: str = "tf220-inference",
):
    """Create Model + EndpointConfig + Endpoint and wait for InService.

    Returns (endpoint, endpoint_name, model_name).

    Deliberately scope-agnostic: the caller owns the pytest fixture scope and
    the try/finally. Each resource is appended to the caller's ``resources``
    list as soon as it is created, so a mid-flight failure still leaves the
    caller able to tear down whatever already exists (prevents billing leaks).
    Tear down with ``_cleanup(reversed(resources))`` — SageMaker requires
    endpoint before endpoint-config before model.
    """
    from sagemaker.core.resources import (
        ContainerDefinition,
        Endpoint,
        EndpointConfig,
        Model,
        ProductionVariant,
    )

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
        execution_role_arn=role_arn,
        session=session,
    )
    resources.append(model)

    variant_kwargs = dict(
        variant_name="AllTraffic",
        model_name=model_name,
        initial_instance_count=1,
        instance_type=sm_instance_type,
    )
    if sm_device_type == "gpu":
        variant_kwargs["inference_ami_version"] = INFERENCE_AMI_VERSION_CU12

    endpoint_config = EndpointConfig.create(
        endpoint_config_name=endpoint_name,
        production_variants=[ProductionVariant(**variant_kwargs)],
        session=session,
    )
    resources.append(endpoint_config)

    endpoint = Endpoint.create(
        endpoint_name=endpoint_name,
        endpoint_config_name=endpoint_name,
        session=session,
    )
    resources.append(endpoint)

    endpoint.wait_for_status("InService")
    return endpoint, endpoint_name, model_name


@pytest.fixture
def deploy_endpoint(
    aws_session,
    sagemaker_session,
    image_uri,
    sm_instance_type,
    sm_device_type,
):
    """Deploy a SageMaker endpoint; yields (endpoint, endpoint_name, model_name).

    Uses try/finally so partially-created resources are always torn down —
    even if deployment fails mid-flight (prevents billing leaks).
    """
    session = aws_session.session
    role_arn = aws_session.resolve_role_arn(SAGEMAKER_ROLE)
    resources: list = []

    def _deploy(
        *,
        model_data_url: str,
        mode: str = "SingleModel",
        container_env: dict | None = None,
        name_prefix: str = "tf220-inference",
    ):
        return _provision_endpoint(
            resources=resources,
            session=session,
            role_arn=role_arn,
            image_uri=image_uri,
            sm_instance_type=sm_instance_type,
            sm_device_type=sm_device_type,
            model_data_url=model_data_url,
            mode=mode,
            container_env=container_env,
            name_prefix=name_prefix,
        )

    try:
        yield _deploy
    finally:
        _cleanup(reversed(resources))
