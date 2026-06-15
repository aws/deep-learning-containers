"""Single-model endpoint integration test for TF 2.20 inference DLC.

Builds a tiny ``y = 2x`` SavedModel, deploys it to a single-instance SageMaker
endpoint backed by the v2 inference image under test, and asserts the
predicted values.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from .resources.build_sample_model import build_sample_model

INSTANCE_TYPE = "ml.c5.xlarge"


def test_single_model_predict(
    sagemaker_session,
    sagemaker_role_arn,
    inference_image_uri,
    unique_name,
    cleanup_endpoint,
):
    from sagemaker.tensorflow.serving import TensorFlowModel

    with tempfile.TemporaryDirectory(prefix="tf220-single-") as workdir:
        tar_path = build_sample_model(
            output_dir=workdir,
            multiplier=2.0,
            model_name="model",
        )

        bucket = sagemaker_session.default_bucket()
        key_prefix = f"tf220-inference-tests/{Path(tar_path).stem}-{unique_name('single')}"
        model_data = sagemaker_session.upload_data(
            path=tar_path,
            bucket=bucket,
            key_prefix=key_prefix,
        )

        endpoint_name = unique_name("tf220-single")
        model_name = unique_name("tf220-single-model")
        cleanup_endpoint(endpoint_name, model_name=model_name)

        tf_model = TensorFlowModel(
            model_data=model_data,
            role=sagemaker_role_arn,
            image_uri=inference_image_uri,
            sagemaker_session=sagemaker_session,
            name=model_name,
        )

        predictor = tf_model.deploy(
            initial_instance_count=1,
            instance_type=INSTANCE_TYPE,
            endpoint_name=endpoint_name,
        )

        try:
            payload = {"instances": [[1.0, 2.0, 3.0]]}
            response = predictor.predict(payload)

            # The TFS predictor may return a dict already, or a JSON string —
            # normalize both shapes.
            if isinstance(response, (bytes, str)):
                response = json.loads(response)

            assert "predictions" in response, f"missing predictions key in {response!r}"
            predictions = response["predictions"]
            assert predictions and isinstance(predictions, list)

            # Output signature is {"output": x * 2.0} -> TFS surfaces the tensor
            # under the signature output key when there is a single named tensor;
            # some TFS versions instead return the raw list. Handle both.
            first = predictions[0]
            if isinstance(first, dict) and "output" in first:
                values = first["output"]
            else:
                values = first

            assert values == pytest.approx([2.0, 4.0, 6.0]), f"got {values!r}"
        finally:
            # cleanup_endpoint teardown handles resources; no manual delete needed.
            pass
