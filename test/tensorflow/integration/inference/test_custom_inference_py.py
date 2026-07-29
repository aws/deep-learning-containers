"""Customer-supplied inference.py handler test for TF 2.20 inference DLC.

Deploys a SavedModel packaged with a customer ``code/inference.py`` that
implements the standard ``input_handler`` + ``output_handler`` contract.
Asserts both handlers fired end-to-end: the input handler prepends a marker
sample, and the output handler adds a ``_handler_marker`` key to the response.

Covers audit finding G2 — the #1 customer usage pattern for TF inference,
uncovered on master TF 2.19 by 5 handler variants, uncovered here by 0
tests until this file.
"""

from __future__ import annotations

import json
import tempfile

import pytest

from .resources.build_sample_model import build_sample_model
from .resources.helpers import CUSTOM_INFERENCE_PY, upload_tarball


def test_custom_inference_py_handlers(
    sagemaker_session,
    deploy_endpoint,
    unique_name,
    cleanup_endpoint,
):
    with tempfile.TemporaryDirectory(prefix="tf220-inference-py-") as workdir:
        tar_path = build_sample_model(
            output_dir=workdir,
            multiplier=2.0,
            code_files={"inference.py": CUSTOM_INFERENCE_PY},
        )
        model_data = upload_tarball(
            sagemaker_session,
            tar_path,
            key_prefix=f"tf220-inference-tests/inference-py/{unique_name('run')}",
        )

        endpoint, endpoint_name, model_name = deploy_endpoint(
            model_data_url=model_data,
            name_prefix="tf220-inference-py",
        )
        cleanup_endpoint(endpoint_name, model_name=model_name)

        # Customer sends 2 rows; input_handler prepends 1 marker row → 3 rows
        # returned by TFS; output_handler wraps the response with a marker key.
        payload = json.dumps({"instances": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]})
        result = endpoint.invoke(
            body=payload,
            content_type="application/json",
            accept="application/json",
        )
        body = json.loads(result.body.read().decode("utf-8"))

        # output_handler marker — proves the customer output_handler ran.
        assert body.get("_handler_marker") == "input_output_ok", (
            f"output_handler marker missing from response: {body!r}"
        )

        # 3 predictions (1 marker + 2 customer rows) — proves input_handler
        # prepended a row and TFS multiplied each element by 2.0.
        predictions = body["predictions"]
        assert len(predictions) == 3, (
            f"expected 3 predictions (1 marker + 2 customer), got {len(predictions)}: "
            f"{predictions!r}"
        )

        # Marker row [9,9,9] * 2 = [18,18,18] (idx 0).
        marker_values = (
            predictions[0]["output"]
            if isinstance(predictions[0], dict) and "output" in predictions[0]
            else predictions[0]
        )
        assert marker_values == pytest.approx([18.0, 18.0, 18.0]), (
            f"marker row got {marker_values!r}"
        )
