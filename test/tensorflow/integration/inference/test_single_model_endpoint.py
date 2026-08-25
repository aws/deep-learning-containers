"""Single-model endpoint integration test for TF 2.20 inference DLC.

Builds a tiny y=2x SavedModel, deploys to a single-instance SM endpoint,
asserts predicted values via SDK v3 sagemaker-core resource layer.
"""

from __future__ import annotations

import json
import tempfile

import pytest
from test_utils import random_suffix_name

from .resources.build_sample_model import build_sample_model
from .resources.helpers import read_predictions, upload_tarball


def test_single_model_predict(
    sagemaker_session,
    deploy_endpoint,
):
    with tempfile.TemporaryDirectory(prefix="tf220-single-") as workdir:
        tar_path = build_sample_model(
            output_dir=workdir,
            multiplier=2.0,
        )

        model_data = upload_tarball(
            sagemaker_session,
            tar_path,
            key_prefix=f"tf220-inference-tests/single/{random_suffix_name('run', 63)}",
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
        values = read_predictions(result)
        assert values == pytest.approx([2.0, 4.0, 6.0]), f"got {values!r}"
