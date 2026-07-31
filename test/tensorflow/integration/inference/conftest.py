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


# SageMaker instance-type map keyed on the image's device_type (from the config
# metadata forwarded via SM_DEVICE_TYPE by the reusable workflow). Kept as
# module-level constants so a test can `import` them for parametrization; the
# `sm_instance_type` fixture is the normal entry point.
#
# GPU is ml.g6.4xlarge to match the CI account's provisioned hosting fleet.
# ml.g5.xlarge is NOT hosted in this account — CreateEndpoint on it fails with
# a generic `CannotStartContainerError` and zero container stdout (docker /
# nvidia-runtime layer, before entrypoint). Verified 2026-07-29: same image
# on ml.g6.4xlarge reaches InService cleanly. Matches TEI's working GPU
# sagemaker-test (test/tei/sagemaker/sagemaker_dlc_test.py); other v2
# frameworks use the same family (vLLM ml.g6.xlarge, openfold3 ml.g6.12xlarge).
_SM_INSTANCE_TYPE_BY_DEVICE = {
    "cpu": "ml.c5.xlarge",
    "gpu": "ml.g6.4xlarge",
}


@pytest.fixture(scope="session")
def sm_instance_type() -> str:
    """SageMaker endpoint instance type appropriate for the image under test.

    Selected from the ``SM_DEVICE_TYPE`` env var (``cpu``/``gpu``) forwarded
    from the reusable workflow. Defaults to CPU when the env var is absent so
    ``pytest`` from a developer laptop keeps working. The GPU mapping
    (``ml.g6.4xlarge``) is what actually exercises the CUDA image's cuDNN /
    tensorflow_model_server GPU path end-to-end.
    """
    device = os.environ.get("SM_DEVICE_TYPE", "cpu").lower()
    return _SM_INSTANCE_TYPE_BY_DEVICE.get(device, "ml.c5.xlarge")


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
def deploy_endpoint(
    boto_session,
    sagemaker_session,
    sagemaker_role_arn,
    inference_image_uri,
    sm_instance_type,
    unique_name,
    cleanup_endpoint,
):
    """Deploy a SageMaker endpoint and return a callable that returns the
    ``Endpoint`` handle plus its name — pair with ``cleanup_endpoint`` for
    teardown. Wraps the ``Model.create -> EndpointConfig.create ->
    Endpoint.create -> wait_for_status`` sequence used by every integration
    test so individual tests stay focused on assertions.

    Cleanup registration happens BEFORE the first AWS create call. If any
    step (``Model.create``, ``EndpointConfig.create``, ``Endpoint.create``,
    or ``wait_for_status``) raises, the endpoint config / partial endpoint /
    model that was created still gets torn down at fixture teardown — a
    ``wait_for_status`` failure previously never returned to the caller,
    so the call-site ``cleanup_endpoint(...)`` line never ran and the
    endpoint billed until the account was scrubbed. Both this fixture and
    ``cleanup_endpoint`` are function-scoped, so the injection lifetime
    matches; cleanup calls at test call sites are now redundant no-ops
    (double-register is safe — teardown swallows NotFound / already-deleted
    exceptions per resource).

    Usage::

        def test_x(deploy_endpoint):
            endpoint, endpoint_name, model_name = deploy_endpoint(
                model_data_url="s3://.../model.tar.gz",  # or an MME prefix
                mode="SingleModel",                       # or "MultiModel"
                container_env={"SAGEMAKER_TFS_ENABLE_BATCHING": "true"},
                name_prefix="tf220-batching",
            )
            ...
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

        # Register cleanup BEFORE any AWS mutation so a mid-flight failure
        # (Model.create, EndpointConfig.create, Endpoint.create, or the
        # wait_for_status poll) still gets torn down at fixture teardown.
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
