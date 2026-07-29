"""Shared helpers for TF 2.20 inference integration tests.

Kept small on purpose — heavy lifting stays in ``conftest.py`` (session
fixtures) and ``build_sample_model.py`` (SavedModel construction). This
module exists so multiple test files can share the S3-upload boilerplate
and the customer ``inference.py`` fixture text without duplicating strings.
"""

from __future__ import annotations


# Minimal customer inference.py used by G2. Implements the standard
# input_handler / output_handler contract that python_service.py imports at
# request time (see python_service.py::_import_handlers). Tests deploy a
# tarball with this file under code/inference.py and assert both handlers
# fired (input_handler injects a marker instance; output_handler appends a
# marker key to the response).
CUSTOM_INFERENCE_PY = '''\
"""Test-only customer inference.py — exercises the SageMaker TFS handler
contract (input_handler + output_handler). If the DLC handler ever stops
importing customer code from /opt/ml/model/code/inference.py, this file
never runs and the assertions in test_custom_inference_py.py fail."""

import json


def input_handler(data, context):
    """Prepend a marker sample so the response reflects handler invocation."""
    payload = data.read().decode("utf-8") if hasattr(data, "read") else data
    body = json.loads(payload)
    instances = body.get("instances", [])
    # Marker: a fixed leading instance we can identify in the response.
    marker = [[9.0, 9.0, 9.0]]
    body["instances"] = marker + instances
    return json.dumps(body)


def output_handler(response, context):
    """Wrap the TFS response with a marker key so the test can prove this
    ran end-to-end. Returns (body, content_type) as required by the DLC."""
    body = json.loads(response.content.decode("utf-8"))
    body["_handler_marker"] = "input_output_ok"
    return json.dumps(body), context.accept_header
'''


def upload_tarball(sagemaker_session, tar_path: str, key_prefix: str) -> str:
    """Upload a model tarball to the SageMaker session's default bucket and
    return the resulting ``s3://`` URL. Thin wrapper around ``upload_data`` —
    exists mostly so test files don't repeat the two-line dance."""
    bucket = sagemaker_session.default_bucket()
    return sagemaker_session.upload_data(
        path=str(tar_path),
        bucket=bucket,
        key_prefix=key_prefix,
    )


def read_predictions(invoke_result) -> list:
    """Pull the numeric output list out of an InvokeEndpointOutput body.

    Handles both signature-keyed responses (``[{"output": [...]}]``) and raw
    row responses that some TFS versions surface. Raises AssertionError if
    the shape is unexpected — tests reading only the numbers can call this
    and skip the shape branching."""
    import json

    body = json.loads(invoke_result.body.read().decode("utf-8"))
    assert "predictions" in body, f"missing predictions key in {body!r}"
    predictions = body["predictions"]
    assert predictions and isinstance(predictions, list)
    first = predictions[0]
    if isinstance(first, dict) and "output" in first:
        return first["output"]
    return first
