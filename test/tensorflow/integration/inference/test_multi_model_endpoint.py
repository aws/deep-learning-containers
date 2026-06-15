"""Multi-model endpoint (MME) integration test for TF 2.20 inference DLC.

Builds two tiny SavedModels (``y = 2x`` and ``y = 3x``), uploads both to a
shared S3 prefix, deploys a SageMaker MME backed by the v2 inference image,
and asserts that ``TargetModel`` routes invocations correctly.

Uses boto3 directly (sagemaker, sagemaker-runtime, s3 clients) rather than
the SageMaker Python SDK, because the v3.x ``sagemaker`` package removed
the v2 ``MultiDataModel`` / ``TensorFlowModel`` classes this test originally
relied on. MME is a thin server-side construct anyway: ``Mode: MultiModel``
on the PrimaryContainer plus a model_data S3 prefix.
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


def _values_from_predictions(predictions) -> list:
    """Pull the numeric output list out of either signature-keyed or raw rows."""
    assert predictions and isinstance(predictions, list)
    first = predictions[0]
    if isinstance(first, dict) and "output" in first:
        return first["output"]
    return first


def test_mme_two_models(
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
    with tempfile.TemporaryDirectory(prefix="tf220-mme-") as workdir:
        workdir_path = Path(workdir)

        # Build two models with different multipliers, each in its own subdir
        # so build_sample_model doesn't collide on the SavedModel layout.
        model1_dir = workdir_path / "m1"
        model2_dir = workdir_path / "m2"
        model1_tar = build_sample_model(
            output_dir=model1_dir, multiplier=2.0, model_name="model", tar_filename="model1.tar.gz"
        )
        model2_tar = build_sample_model(
            output_dir=model2_dir, multiplier=3.0, model_name="model", tar_filename="model2.tar.gz"
        )

        run_id = unique_name("mme")
        s3_key_prefix = f"tf220-inference-tests/mme-models/{run_id}"

        # Upload each tarball under the shared MME prefix so the runtime can
        # resolve TargetModel relative to the same S3 location.
        upload_to_s3(model1_tar, default_bucket, f"{s3_key_prefix}/model1.tar.gz")
        upload_to_s3(model2_tar, default_bucket, f"{s3_key_prefix}/model2.tar.gz")
        s3_model_prefix = f"s3://{default_bucket}/{s3_key_prefix}/"

        endpoint_name = unique_name("tf220-mme")
        base_model_name = unique_name("tf220-mme-model")
        cleanup_endpoint(endpoint_name, model_name=base_model_name)

        # ``Mode: MultiModel`` plus an S3 prefix is the entire MME contract on
        # the control plane; the runtime resolves ``TargetModel`` relative to
        # ModelDataUrl.
        sagemaker_client.create_model(
            ModelName=base_model_name,
            ExecutionRoleArn=sagemaker_role_arn,
            PrimaryContainer={
                "Image": inference_image_uri,
                "ModelDataUrl": s3_model_prefix,
                "Mode": "MultiModel",
            },
        )

        sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": base_model_name,
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

        # Invoke model1 (x * 2.0)
        resp1 = sagemaker_runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            TargetModel="model1.tar.gz",
            Body=payload,
        )
        body1 = _decode(resp1)
        assert "predictions" in body1, f"model1 response missing predictions: {body1!r}"
        values1 = _values_from_predictions(body1["predictions"])
        assert values1 == pytest.approx([2.0, 4.0, 6.0]), f"model1 got {values1!r}"

        # Invoke model2 (x * 3.0)
        resp2 = sagemaker_runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            TargetModel="model2.tar.gz",
            Body=payload,
        )
        body2 = _decode(resp2)
        assert "predictions" in body2, f"model2 response missing predictions: {body2!r}"
        values2 = _values_from_predictions(body2["predictions"])
        assert values2 == pytest.approx([3.0, 6.0, 9.0]), f"model2 got {values2!r}"
