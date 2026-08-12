"""Error-path integration tests for TF 2.20 inference DLC.

Verifies the endpoint returns useful 4xx/5xx for malformed input rather than
hanging or leaking nginx HTML. All scenarios share one endpoint (SM endpoint
provisioning dominates cost; scenarios are read-only invocations).
"""

from __future__ import annotations

import json
import tempfile

import pytest
from botocore.exceptions import ClientError
from test_utils import random_suffix_name

from .resources.build_sample_model import build_sample_model
from .resources.helpers import upload_tarball


@pytest.fixture(scope="module")
def error_endpoint(sagemaker_session, aws_session, sagemaker_role_arn, image_uri, sm_instance_type):
    """Deploy a single endpoint shared across all error-path parametrized tests."""
    import logging

    from sagemaker.core.resources import (
        ContainerDefinition,
        Endpoint,
        EndpointConfig,
        Model,
        ProductionVariant,
    )

    LOGGER = logging.getLogger(__name__)
    session = aws_session.session

    with tempfile.TemporaryDirectory(prefix="tf220-errors-") as workdir:
        tar_path = build_sample_model(output_dir=workdir, multiplier=2.0)
        model_data = upload_tarball(
            sagemaker_session,
            tar_path,
            key_prefix=f"tf220-inference-tests/errors/{random_suffix_name('run', 63)}",
        )

        endpoint_name = random_suffix_name("tf220-errors", 63)
        model_name = random_suffix_name("tf220-errors-model", 63)

        Model.create(
            model_name=model_name,
            primary_container=ContainerDefinition(
                image=image_uri,
                model_data_url=model_data,
            ),
            execution_role_arn=sagemaker_role_arn,
            session=session,
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
            session=session,
        )
        endpoint = Endpoint.create(
            endpoint_name=endpoint_name,
            endpoint_config_name=endpoint_name,
            session=session,
        )
        endpoint.wait_for_status("InService")

        try:
            yield endpoint
        finally:
            for resource_cls, kwargs in [
                (Endpoint, {"endpoint_name": endpoint_name}),
                (EndpointConfig, {"endpoint_config_name": endpoint_name}),
                (Model, {"model_name": model_name}),
            ]:
                try:
                    resource_cls.get(session=session, **kwargs).delete()
                except Exception as e:
                    LOGGER.warning(f"Cleanup {resource_cls.__name__} failed: {e}")


def _status(err: ClientError) -> int:
    return err.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)


def _body(err: ClientError) -> str:
    body = err.response.get("Body")
    if hasattr(body, "read"):
        body = body.read()
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    return body or ""


@pytest.mark.parametrize(
    "scenario_id, body, content_type",
    [
        ("malformed-json", b"{ this is not valid json", "application/json"),
        ("empty-body", b"", "application/json"),
        ("unsupported-content-type", b"anything at all", "application/x-unsupported-mimetype"),
        (
            "wrong-tensor-shape",
            json.dumps({"instances": "not_a_list_of_lists"}).encode("utf-8"),
            "application/json",
        ),
    ],
    ids=["malformed-json", "empty-body", "unsupported-content-type", "wrong-tensor-shape"],
)
def test_error_scenario(error_endpoint, scenario_id, body, content_type):
    """Each scenario asserts the endpoint returns 4xx/5xx, not 200 or raw nginx HTML."""
    with pytest.raises(ClientError) as excinfo:
        error_endpoint.invoke(
            body=body,
            content_type=content_type,
            accept="application/json",
        )
    status = _status(excinfo.value)
    assert 400 <= status < 600, f"[{scenario_id}] expected 4xx/5xx, got {status}"

    response_body = _body(excinfo.value)
    if response_body:
        assert "<html" not in response_body.lower(), (
            f"[{scenario_id}] endpoint leaked nginx HTML: {response_body[:200]!r}"
        )
