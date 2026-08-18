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
from test_utils.constants import SAGEMAKER_ROLE

from .conftest import _cleanup, _provision_endpoint
from .resources.build_sample_model import build_sample_model
from .resources.helpers import upload_tarball


@pytest.fixture(scope="module")
def error_endpoint(sagemaker_session, aws_session, image_uri, sm_instance_type, sm_device_type):
    """Deploy a single endpoint shared across all error-path parametrized tests.

    Module-scoped (endpoint provisioning dominates cost), so it cannot reuse the
    function-scoped deploy_endpoint fixture — but it shares the same underlying
    _provision_endpoint helper so endpoint setup lives in exactly one place.
    """
    resources: list = []

    with tempfile.TemporaryDirectory(prefix="tf220-errors-") as workdir:
        tar_path = build_sample_model(
            output_dir=workdir,
            multiplier=2.0,
            code_files={
                "inference.py": (
                    "def input_handler(data, context):\n"
                    "    return data.read().decode('utf-8')\n"
                    "\n"
                    "def output_handler(response, context):\n"
                    "    return response.content, context.accept_header\n"
                ),
            },
        )
        model_data = upload_tarball(
            sagemaker_session,
            tar_path,
            key_prefix=f"tf220-inference-tests/errors/{random_suffix_name('run', 63)}",
        )

        try:
            endpoint, _endpoint_name, _model_name = _provision_endpoint(
                resources=resources,
                session=aws_session.session,
                role_arn=aws_session.resolve_role_arn(SAGEMAKER_ROLE),
                image_uri=image_uri,
                sm_instance_type=sm_instance_type,
                sm_device_type=sm_device_type,
                model_data_url=model_data,
                name_prefix="tf220-errors",
            )
            yield endpoint
        finally:
            _cleanup(reversed(resources))


def _original_status(err: ClientError) -> int:
    """The container's HTTP status. Falls back to outer status if container never saw the request."""
    original = err.response.get("OriginalStatusCode")
    if original is not None:
        return int(original)
    return err.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)


def _original_message(err: ClientError) -> str:
    """The container's response body from OriginalMessage."""
    return err.response.get("OriginalMessage", "")


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
    """Each scenario asserts the container returns 4xx/5xx, not 200 or nginx HTML."""
    with pytest.raises(ClientError) as excinfo:
        error_endpoint.invoke(
            body=body,
            content_type=content_type,
            accept="application/json",
        )
    status = _original_status(excinfo.value)
    assert 400 <= status < 600, (
        f"[{scenario_id}] expected 4xx/5xx, got {status}: {excinfo.value.response!r}"
    )

    response_body = _original_message(excinfo.value)
    assert "<html" not in response_body.lower(), (
        f"[{scenario_id}] endpoint leaked nginx HTML: {response_body[:200]!r}"
    )
