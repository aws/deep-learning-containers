"""Content-Type / Accept negotiation tests for TF 2.20 inference DLC.

Verifies the nginx-njs handler at ``scripts/docker/tensorflow/inference/
sagemaker/tensorflowServing.js`` correctly converts non-JSON payloads to
TFS's REST API JSON shape. Master TF 2.19 tested this via
``test_predict_csv*``; covers audit finding G4.

Focuses on ``text/csv`` — the most common non-JSON input for TF inference.
The njs ``csv_request`` handler at line 191 converts multi-column CSV rows
into a ``{"instances": [[...], [...]]}`` body before forwarding to TFS.
"""

from __future__ import annotations

import json
import tempfile

import pytest

from .resources.build_sample_model import build_sample_model
from .resources.helpers import upload_tarball


def test_csv_content_type_multi_column(
    sagemaker_session,
    deploy_endpoint,
    unique_name,
    cleanup_endpoint,
):
    """Two rows, three numeric columns each — the njs handler must convert
    this into ``{"instances": [[1.0,2.0,3.0], [4.0,5.0,6.0]]}`` before TFS
    sees it. Assertions on the output prove the CSV path is intact."""
    with tempfile.TemporaryDirectory(prefix="tf220-csv-") as workdir:
        tar_path = build_sample_model(output_dir=workdir, multiplier=2.0)
        model_data = upload_tarball(
            sagemaker_session,
            tar_path,
            key_prefix=f"tf220-inference-tests/csv/{unique_name('run')}",
        )
        endpoint, endpoint_name, model_name = deploy_endpoint(
            model_data_url=model_data,
            name_prefix="tf220-csv",
        )
        cleanup_endpoint(endpoint_name, model_name=model_name)

        csv_payload = b"1.0,2.0,3.0\n4.0,5.0,6.0\n"
        result = endpoint.invoke(
            body=csv_payload,
            content_type="text/csv",
            accept="application/json",
        )
        # Read the streaming body ONCE, then parse — reading twice returns
        # an empty bytes object because the underlying stream is exhausted.
        body = json.loads(result.body.read().decode("utf-8"))
        rows = body["predictions"]
        assert len(rows) == 2, f"expected 2 rows from CSV input, got {len(rows)}: {rows!r}"

        def _values(row):
            return row["output"] if isinstance(row, dict) and "output" in row else row

        # Row 1: [1,2,3] * 2 = [2,4,6]; row 2: [4,5,6] * 2 = [8,10,12].
        assert _values(rows[0]) == pytest.approx([2.0, 4.0, 6.0]), (
            f"row 1 got {_values(rows[0])!r}"
        )
        assert _values(rows[1]) == pytest.approx([8.0, 10.0, 12.0]), (
            f"row 2 got {_values(rows[1])!r}"
        )
