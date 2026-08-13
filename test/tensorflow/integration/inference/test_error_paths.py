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
from test_utils.constants import INFERENCE_AMI_VERSION_CU12, SAGEMAKER_ROLE

from .conftest import _cleanup
from .resources.build_sample_model import build_sample_model
from .resources.helpers import upload_tarball


@pytest.fixture(scope="module")
def error_endpoint(sagemaker_session, aws_session, image_uri, sm_instance_type, sm_device_type):
    """Deploy a single endpoint shared across all error-path parametrized tests."""
    from sagemaker.core.resources import (
        ContainerDefinition,
        Endpoint,
        EndpointConfig,
        Model,
        ProductionVariant,
    )

    session = aws_session.session

    with tempfile.TemporaryDirectory(prefix="tf220-errors-") as workdir:
        tar_path = build_sample_model(
            output_dir=workdir,
            multiplier=2.0,
            code_files={
                "inference.py": "def input_handler(data, context):\n"
                "    return data.read().decode('utf-8')\n"
            },
        )
        model_data = upload_tarball(
            sagemaker_session,
            tar_path,
            key_prefix=f"tf220-inference-tests/errors/{random_suffix_name('run', 63)}",
        )

        endpoint_name = random_suffix_name("tf220-errors", 63)
        model_name = random_suffix_name("tf220-errors-model", 63)

        model_obj = endpoint_config = endpoint = None
        try:
            model_obj = Model.create(
                model_name=model_name,
                primary_container=ContainerDefinition(
                    image=image_uri,
                    model_data_url=model_data,
                ),
                execution_role_arn=aws_session.resolve_role_arn(SAGEMAKER_ROLE),
                session=session,
            )
            endpoint_config = EndpointConfig.create(
                endpoint_config_name=endpoint_name,
                production_variants=[
                    ProductionVariant(
                        variant_name="AllTraffic",
                        model_name=model_name,
                        initial_instance_count=1,
                        instance_type=sm_instance_type,
                        **(
                            {"inference_ami_version": INFERENCE_AMI_VERSION_CU12}
                            if sm_device_type == "gpu"
                            else {}
                        ),
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
            yield endpoint
        finally:
            _cleanup([endpoint, endpoint_config, model_obj])


def _original_status(err: ClientError) -> int:
    """The container's actual HTTP status (not the SageMaker wrapper status)."""
    return int(err.response.get("OriginalStatusCode", 0))


def _original_message(err: ClientError) -> str:
    """The container's response body from OriginalMessage."""
    return err.response.get("OriginalMessage", "")


@pytest.mark.parametrize(
    "scenario_id, body, content_type, expected_status",
    [
        ("malformed-json", b"{ this is not valid json", "application/json", 400),
        ("empty-body", b"", "application/json", 400),
        ("unsupported-content-type", b"anything at all", "application/x-unsupported-mimetype", 400),
        (
            "wrong-tensor-shape",
            json.dumps({"instances": "not_a_list_of_lists"}).encode("utf-8"),
            "application/json",
            400,
        ),
    ],
    ids=["malformed-json", "empty-body", "unsupported-content-type", "wrong-tensor-shape"],
)
def test_error_scenario(error_endpoint, scenario_id, body, content_type, expected_status):
    """Each scenario asserts the container returns the expected error status."""
    with pytest.raises(ClientError) as excinfo:
        error_endpoint.invoke(
            body=body,
            content_type=content_type,
            accept="application/json",
        )
    status = _original_status(excinfo.value)
    assert status == expected_status, f"[{scenario_id}] expected {expected_status}, got {status}"

    response_body = _original_message(excinfo.value)
    assert "<html" not in response_body.lower(), (
        f"[{scenario_id}] endpoint leaked nginx HTML: {response_body[:200]!r}"
    )
