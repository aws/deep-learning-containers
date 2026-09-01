"""Shared helpers for TF 2.20 inference integration tests."""

from __future__ import annotations

# Customer inference.py implementing input_handler + output_handler.
# input_handler prepends a marker instance; output_handler appends a
# _handler_marker key. Tests assert both markers appear in the response.
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
    """Upload a model tarball to the SM session's default bucket; return s3:// URL."""
    bucket = sagemaker_session.default_bucket()
    return sagemaker_session.upload_data(
        path=str(tar_path),
        bucket=bucket,
        key_prefix=key_prefix,
    )


def read_predictions(invoke_result) -> list:
    """Pull the numeric output list out of an InvokeEndpointOutput body.

    Handles three TFS output shapes: [{"output": [...]}] (build_sample_model
    signature), [{"output_0": [...]}] (Keras 3 model.export default name),
    and [[...]] (some TFS versions flatten single-output signatures).
    """
    import json

    body = json.loads(invoke_result.body.read().decode("utf-8"))
    assert "predictions" in body, f"missing predictions key in {body!r}"
    predictions = body["predictions"]
    assert predictions and isinstance(predictions, list)
    first = predictions[0]
    if isinstance(first, dict):
        # Prefer the historically-used ``output`` key.
        if "output" in first:
            return first["output"]
        # Fall back to any single-key dict (e.g. Keras 3's ``output_0``).
        assert len(first) == 1, f"expected single-key output dict, got keys {list(first)!r}"
        return next(iter(first.values()))
    return first
