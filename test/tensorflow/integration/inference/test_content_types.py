"""Content-Type negotiation tests for TF 2.20 inference DLC.

Focuses on text/csv — the njs csv_request handler converts CSV rows into a
{"instances": [[...], [...]]} body before forwarding to TFS.
"""

from __future__ import annotations

import json
import tempfile

import pytest

from .resources.build_sample_model import build_sample_model
from .resources.helpers import upload_tarball
from test_utils import random_suffix_name


def test_csv_content_type_multi_column(
    sagemaker_session,
    deploy_endpoint,
    cleanup_endpoint,
):
    """Two rows, three numeric columns each; assert 2x output."""
    with tempfile.TemporaryDirectory(prefix="tf220-csv-") as workdir:
        tar_path = build_sample_model(output_dir=workdir, multiplier=2.0)
        model_data = upload_tarball(
            sagemaker_session,
            tar_path,
            key_prefix=f"tf220-inference-tests/csv/{random_suffix_name('run', 63)}",
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
        # Read the streaming body ONCE — a second read returns empty bytes.
        body = json.loads(result.body.read().decode("utf-8"))
        rows = body["predictions"]
        assert len(rows) == 2, f"expected 2 rows from CSV input, got {len(rows)}: {rows!r}"

        def _values(row):
            return row["output"] if isinstance(row, dict) and "output" in row else row

        # Row 1: [1,2,3] * 2 = [2,4,6]; row 2: [4,5,6] * 2 = [8,10,12].
        assert _values(rows[0]) == pytest.approx([2.0, 4.0, 6.0]), f"row 1 got {_values(rows[0])!r}"
        assert _values(rows[1]) == pytest.approx([8.0, 10.0, 12.0]), (
            f"row 2 got {_values(rows[1])!r}"
        )
