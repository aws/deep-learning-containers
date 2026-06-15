"""Single-model endpoint integration test for TF 2.20 inference DLC.

Builds a tiny ``y = 2x`` SavedModel, deploys it to a single-instance SageMaker
endpoint backed by the v2 inference image under test, and asserts the
predicted values.

Uses boto3 directly (sagemaker, sagemaker-runtime, s3 clients) rather than
the SageMaker Python SDK, because the v3.x ``sagemaker`` package removed the
v2 ``sagemaker.tensorflow.serving.TensorFlowModel`` flow this test originally
relied on.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from .resources.build_sample_model import build_sample_model

INSTANCE_TYPE = "ml.c5.xlarge"


def _decode(invoke_response) -> dict:
    """Decode an ``invoke_endpoint`` response Body into a Python dict."""
    body = invoke_response["Body"].read()
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8")
    return json.loads(body)


def test_single_model_predict(
    sagemaker_client,
    sagemaker_runtime_client,
    sagemaker_role_arn,
    inference_image_uri,
    default_bucket,
    upload_to_s3,
    unique_name,
    cleanup_endpoint,
    wait_for_endpoint,
):
    with tempfile.TemporaryDirectory(prefix="tf220-single-") as workdir:
        tar_path = build_sample_model(
            output_dir=workdir,
            multiplier=2.0,
            model_name="model",
        )

        run_id = unique_name("single")
        s3_key = f"tf220-inference-tests/{Path(tar_path).stem}-{run_id}/{Path(tar_path).name}"
        model_data = upload_to_s3(tar_path, default_bucket, s3_key)

        endpoint_name = unique_name("tf220-single")
        model_name = unique_name("tf220-single-model")
        cleanup_endpoint(endpoint_name, model_name=model_name)

        # Equivalent to v2 ``TensorFlowModel(image_uri=..., model_data=...).deploy(...)``
        # but expressed as the underlying control-plane API calls.
        sagemaker_client.create_model(
            ModelName=model_name,
            ExecutionRoleArn=sagemaker_role_arn,
            PrimaryContainer={
                "Image": inference_image_uri,
                "ModelDataUrl": model_data,
            },
        )

        sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InitialInstanceCount": 1,
                    "InstanceType": INSTANCE_TYPE,
                    "InitialVariantWeight": 1.0,
                }
            ],
        )

        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_name,
        )
        wait_for_endpoint(endpoint_name)

        payload = json.dumps({"instances": [[1.0, 2.0, 3.0]]})
        response = sagemaker_runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=payload,
        )
        body = _decode(response)

        assert "predictions" in body, f"missing predictions key in {body!r}"
        predictions = body["predictions"]
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
