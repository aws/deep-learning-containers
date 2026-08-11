"""Single-model endpoint integration test for TF 2.20 inference DLC.

Builds a tiny y=2x SavedModel, deploys to a single-instance SM endpoint,
asserts predicted values via SDK v3 sagemaker-core resource layer.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from .resources.build_sample_model import build_sample_model
from test_utils import random_suffix_name


def test_single_model_predict(
    sagemaker_session,
    deploy_endpoint,
):
    with tempfile.TemporaryDirectory(prefix="tf220-single-") as workdir:
        tar_path = build_sample_model(
            output_dir=workdir,
            multiplier=2.0,
        )

        bucket = sagemaker_session.default_bucket()
        key_prefix = f"tf220-inference-tests/{Path(tar_path).stem}-{random_suffix_name('single', 63)}"
        model_data = sagemaker_session.upload_data(
            path=tar_path,
            bucket=bucket,
            key_prefix=key_prefix,
        )

        endpoint, endpoint_name, model_name = deploy_endpoint(
            model_data_url=model_data,
        )

        payload = json.dumps({"instances": [[1.0, 2.0, 3.0]]})
        result = endpoint.invoke(
            body=payload,
            content_type="application/json",
            accept="application/json",
        )
        response = json.loads(result.body.read().decode("utf-8"))

        assert "predictions" in response, f"missing predictions key in {response!r}"
        predictions = response["predictions"]
        assert predictions and isinstance(predictions, list)

        first = predictions[0]
        if isinstance(first, dict) and "output" in first:
            values = first["output"]
        else:
            values = first

        assert values == pytest.approx([2.0, 4.0, 6.0]), f"got {values!r}"
