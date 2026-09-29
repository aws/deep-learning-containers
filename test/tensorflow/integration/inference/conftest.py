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
from test_utils.instance_capacity import (
    build_instance_pools,
    deploy_with_capacity_fallback,
    normalize_instance_types,
)

LOGGER = logging.getLogger(__name__)

# SM instance-type map keyed on the image's device_type. The test models are tiny, so
# any single-GPU rung works; GPU is a priority-ordered ladder across the L4 and A10G
# capacity pools (same ladder as the vLLM/SGLang endpoint tests) to survive ICE.
_SM_INSTANCE_TYPE_BY_DEVICE = {
    "cpu": "ml.c5.xlarge",
    "gpu": ["ml.g6.xlarge", "ml.g6.2xlarge", "ml.g6.4xlarge", "ml.g5.2xlarge", "ml.g5.12xlarge"],
}


@pytest.fixture(scope="session")
def sm_device_type() -> str:
    """Device type from SM_DEVICE_TYPE env var (cpu|gpu). Fail-closed on misconfig."""
    device = os.environ.get("SM_DEVICE_TYPE", "").lower()
    assert device in {"cpu", "gpu"}, f"SM_DEVICE_TYPE must be 'cpu' or 'gpu'; got {device!r}."
    return device


@pytest.fixture(scope="session")
def sm_instance_type(sm_device_type) -> str | list[str]:
    """SM endpoint instance type (or fallback ladder) derived from device type."""
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
    sm_instance_type: str | list[str],
    sm_device_type: str,
    model_data_url: str,
    mode: str = "SingleModel",
    container_env: dict | None = None,
    name_prefix: str = "tf220-inference",
):
    """Create Model + EndpointConfig + Endpoint and wait for InService.

    Returns (endpoint, endpoint_name, model_name).

    ``sm_instance_type`` may be a priority-ordered ladder. SingleModel endpoints get it
    as native SageMaker instance pools (server-side fallback in one deploy). MultiModel
    endpoints walk it client-side, one deploy per rung, since instance pools are not
    documented for multi-model endpoints.

    Deliberately scope-agnostic: the caller owns the pytest fixture scope and
    the try/finally. A failed deploy tears down its own partial resources; a
    successful one appends them to the caller's ``resources`` list. Tear down
    with ``_cleanup(reversed(resources))`` — SageMaker requires endpoint before
    endpoint-config before model.
    """
    from sagemaker.core.resources import (
        ContainerDefinition,
        Endpoint,
        EndpointConfig,
        Model,
    )
    from sagemaker.core.shapes import ProductionVariant

    container_kwargs = {
        "image": image_uri,
        "model_data_url": model_data_url,
    }
    if mode == "MultiModel":
        container_kwargs["mode"] = "MultiModel"
    if container_env:
        container_kwargs["environment"] = dict(container_env)

    def _create(capacity_kwargs):
        endpoint_name = random_suffix_name(name_prefix, 63)
        model_name = random_suffix_name(f"{name_prefix}-model", 63)
        attempt: list = []
        try:
            model = Model.create(
                model_name=model_name,
                primary_container=ContainerDefinition(**container_kwargs),
                execution_role_arn=role_arn,
                session=session,
            )
            attempt.append(model)

            variant_kwargs = dict(
                variant_name="AllTraffic",
                model_name=model_name,
                initial_instance_count=1,
                **capacity_kwargs,
            )
            if sm_device_type == "gpu":
                variant_kwargs["inference_ami_version"] = INFERENCE_AMI_VERSION_CU12

            endpoint_config = EndpointConfig.create(
                endpoint_config_name=endpoint_name,
                production_variants=[ProductionVariant(**variant_kwargs)],
                session=session,
            )
            attempt.append(endpoint_config)

            endpoint = Endpoint.create(
                endpoint_name=endpoint_name,
                endpoint_config_name=endpoint_name,
                session=session,
            )
            attempt.append(endpoint)

            endpoint.wait_for_status("InService")
        except BaseException:
            # Tear down this rung now so a capacity retry does not leak a Failed endpoint.
            _cleanup(reversed(attempt))
            raise
        resources.extend(attempt)
        return endpoint, endpoint_name, model_name

    types = normalize_instance_types(sm_instance_type)
    if len(types) == 1:
        return _create({"instance_type": types[0]})
    if mode == "MultiModel":
        return deploy_with_capacity_fallback(
            types, lambda t: _create({"instance_type": t}), label=name_prefix
        )
    return _create({"instance_pools": build_instance_pools(types)})


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
