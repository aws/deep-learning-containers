"""Error-path integration tests for TF 2.20 inference DLC.

Verifies the endpoint returns useful 4xx/5xx for malformed input rather than
hanging or leaking nginx HTML. Drives all scenarios through one endpoint
(scenarios are read-only; SM endpoint provisioning dominates cost).
"""

from __future__ import annotations

import json
import tempfile

import pytest
from botocore.exceptions import ClientError

from .resources.build_sample_model import build_sample_model
from .resources.helpers import upload_tarball
from test_utils import random_suffix_name

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
        # Must not leak a raw nginx HTML 500 page.
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
    cleanup_endpoint,
):
    """Single endpoint deploy, drive all error scenarios via a loop."""
    with tempfile.TemporaryDirectory(prefix="tf220-errors-") as workdir:
        tar_path = build_sample_model(output_dir=workdir, multiplier=2.0)
        model_data = upload_tarball(
            sagemaker_session,
            tar_path,
            key_prefix=f"tf220-inference-tests/errors/{random_suffix_name('run', 63)}",
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
                    # Empty bodies are OK (SM sometimes strips them).
                    if response_body and not extra_check(response_body):
                        failures.append(
                            f"[{scenario_id}] body failed extra check: {response_body[:200]!r}"
                        )
            except pytest.fail.Exception as e:  # noqa: PERF203
                # Endpoint accepted a payload it should have rejected.
                failures.append(f"[{scenario_id}] endpoint accepted bad input: {e}")

        assert not failures, "error-path scenarios failed:\n  " + "\n  ".join(failures)
