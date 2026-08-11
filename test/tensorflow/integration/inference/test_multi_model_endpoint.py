"""Multi-model endpoint (MME) integration test for TF 2.20 inference DLC.

Builds two tiny SavedModels (y=2x, y=3x), uploads to a shared S3 prefix,
deploys an MME, and asserts target_model routes invocations correctly.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from .resources.build_sample_model import build_sample_model
from test_utils import random_suffix_name


def _values_from_predictions(predictions) -> list:
    """Pull the numeric output list out of either signature-keyed or raw rows."""
    assert predictions and isinstance(predictions, list)
    first = predictions[0]
    if isinstance(first, dict) and "output" in first:
        return first["output"]
    return first


def test_mme_two_models(
    sagemaker_session,
    deploy_endpoint,
):
    with tempfile.TemporaryDirectory(prefix="tf220-mme-") as workdir:
        workdir_path = Path(workdir)

        model1_dir = workdir_path / "m1"
        model2_dir = workdir_path / "m2"
        model1_tar = build_sample_model(
            output_dir=model1_dir, multiplier=2.0, tar_filename="model1.tar.gz"
        )
        model2_tar = build_sample_model(
            output_dir=model2_dir, multiplier=3.0, tar_filename="model2.tar.gz"
        )

        bucket = sagemaker_session.default_bucket()
        run_id = random_suffix_name("mme", 63)
        s3_key_prefix = f"tf220-inference-tests/mme-models/{run_id}"

        sagemaker_session.upload_data(path=model1_tar, bucket=bucket, key_prefix=s3_key_prefix)
        sagemaker_session.upload_data(path=model2_tar, bucket=bucket, key_prefix=s3_key_prefix)
        s3_model_prefix = f"s3://{bucket}/{s3_key_prefix}/"

        endpoint, endpoint_name, model_name = deploy_endpoint(
            model_data_url=s3_model_prefix,
            mode="MultiModel",
            name_prefix="tf220-mme",
        )

        payload = json.dumps({"instances": [[1.0, 2.0, 3.0]]})

        resp1 = endpoint.invoke(
            body=payload,
            content_type="application/json",
            accept="application/json",
            target_model="model1.tar.gz",
        )
        body1 = json.loads(resp1.body.read().decode("utf-8"))
        assert "predictions" in body1, f"model1 response missing predictions: {body1!r}"
        values1 = _values_from_predictions(body1["predictions"])
        assert values1 == pytest.approx([2.0, 4.0, 6.0]), f"model1 got {values1!r}"

        resp2 = endpoint.invoke(
            body=payload,
            content_type="application/json",
            accept="application/json",
            target_model="model2.tar.gz",
        )
        body2 = json.loads(resp2.body.read().decode("utf-8"))
        assert "predictions" in body2, f"model2 response missing predictions: {body2!r}"
        values2 = _values_from_predictions(body2["predictions"])
        assert values2 == pytest.approx([3.0, 6.0, 9.0]), f"model2 got {values2!r}"
