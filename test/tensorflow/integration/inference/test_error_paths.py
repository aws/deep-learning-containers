"""Error-path integration tests for TF 2.20 inference DLC.

Verifies the endpoint returns useful 4xx/5xx responses for malformed or
unsupported client input rather than hanging, crashing, or leaking nginx
default error pages. Covers audit finding G3.

SageMaker Runtime surfaces non-2xx container responses via a botocore
``ClientError`` (subclasses ``ModelError`` / ``ValidationError``). The error
carries the container's response body in ``err.response['Body']`` (bytes
or ``StreamingBody``) and the HTTP status in
``err.response['ResponseMetadata']['HTTPStatusCode']``.

Deploys a single endpoint and drives all error scenarios through it as a
single parametrized test — SageMaker endpoint provisioning is the dominant
cost (~5 min), so amortizing across scenarios matters. All scenarios are
read-only on server state, so sharing is safe.
"""

from __future__ import annotations

import json
import tempfile

import pytest
from botocore.exceptions import ClientError

from .resources.build_sample_model import build_sample_model
from .resources.helpers import upload_tarball


# Scenarios: (scenario_id, body, content_type, extra_body_check).
# ``extra_body_check`` runs on the container's response body string; None
# means "just require non-empty status in the 4xx/5xx range".
ERROR_SCENARIOS = [
    (
        "malformed-json",
        b"{ this is not valid json",
        "application/json",
        None,
    ),
    (
        "empty-body",
        b"",
        "application/json",
        None,
    ),
    (
        "unsupported-content-type",
        b"anything at all",
        "application/x-unsupported-mimetype",
        None,
    ),
    (
        "wrong-tensor-shape",
        json.dumps({"instances": "not_a_list_of_lists"}).encode("utf-8"),
        "application/json",
        # Response must not leak a raw nginx HTML 500 page — customers rely
        # on the JSON error string for debugging.
        lambda body: "<html" not in body.lower(),
    ),
]


def _status(err: ClientError) -> int:
    return err.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)


def _body(err: ClientError) -> str:
    body = err.response.get("Body")
    if hasattr(body, "read"):
        body = body.read()
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    return body or ""


def test_error_scenarios_return_useful_4xx_or_5xx(
    sagemaker_session,
    deploy_endpoint,
    unique_name,
    cleanup_endpoint,
):
    """Single endpoint deploy, drive all error scenarios against it.

    Loops over ``ERROR_SCENARIOS`` and asserts each one. Not parametrized
    across pytest cases because parametrizing would force per-case fixture
    re-invocation and either re-deploy or violate fixture-scope rules; a
    plain loop is the simplest way to share the endpoint safely.
    """
    with tempfile.TemporaryDirectory(prefix="tf220-errors-") as workdir:
        tar_path = build_sample_model(output_dir=workdir, multiplier=2.0)
        model_data = upload_tarball(
            sagemaker_session,
            tar_path,
            key_prefix=f"tf220-inference-tests/errors/{unique_name('run')}",
        )
        endpoint, endpoint_name, model_name = deploy_endpoint(
            model_data_url=model_data,
            name_prefix="tf220-errors",
        )
        cleanup_endpoint(endpoint_name, model_name=model_name)

        failures = []
        for scenario_id, body, content_type, extra_check in ERROR_SCENARIOS:
            try:
                with pytest.raises(ClientError) as excinfo:
                    endpoint.invoke(
                        body=body,
                        content_type=content_type,
                        accept="application/json",
                    )
                status = _status(excinfo.value)
                if not (400 <= status < 600):
                    failures.append(f"[{scenario_id}] expected 4xx/5xx, got status {status}")
                    continue
                if extra_check is not None:
                    response_body = _body(excinfo.value)
                    # Empty bodies are fine — SageMaker Runtime sometimes
                    # returns just a status with no container body attached.
                    # The extra_check exists to guard against actively-bad
                    # bodies (e.g. raw nginx HTML pages), not to require
                    # a body at all.
                    if response_body and not extra_check(response_body):
                        failures.append(
                            f"[{scenario_id}] body failed extra check: {response_body[:200]!r}"
                        )
            except pytest.fail.Exception as e:  # noqa: PERF203
                # pytest.raises didn't raise → the endpoint accepted a
                # payload it should have rejected. That's a real bug.
                failures.append(f"[{scenario_id}] endpoint accepted bad input: {e}")

        assert not failures, "error-path scenarios failed:\n  " + "\n  ".join(failures)
