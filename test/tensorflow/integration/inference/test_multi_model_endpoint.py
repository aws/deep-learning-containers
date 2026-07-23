"""Multi-model endpoint (MME) integration test for TF 2.20 inference DLC.

Builds two tiny SavedModels (``y = 2x`` and ``y = 3x``), uploads both to a
shared S3 prefix, deploys a SageMaker MME backed by the v2 inference image,
and asserts that ``target_model`` routes invocations correctly.

Uses the SageMaker Python SDK v3 ``sagemaker-core`` resource layer — v3
removed ``sagemaker.multidatamodel.MultiDataModel``. The native MME wire
contract (``ContainerDefinition.mode = "MultiModel"``,
``model_data_url = s3://bucket/prefix/``, plus the
``X-Amzn-SageMaker-Target-Model`` header on invoke) is unchanged, so we
express it directly: ``Model.create`` with ``mode="MultiModel"`` and an S3
prefix in ``model_data_url``, then ``endpoint.invoke(target_model=...)``
which sets the runtime header for us.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from .resources.build_sample_model import build_sample_model

INSTANCE_TYPE = "ml.c5.xlarge"


def _values_from_predictions(predictions) -> list:
    """Pull the numeric output list out of either signature-keyed or raw rows."""
    assert predictions and isinstance(predictions, list)
    first = predictions[0]
    if isinstance(first, dict) and "output" in first:
        return first["output"]
    return first


def test_mme_two_models(
    boto_session,
    sagemaker_session,
    sagemaker_role_arn,
    inference_image_uri,
    unique_name,
    cleanup_endpoint,
):
    from sagemaker.core.resources import (
        ContainerDefinition,
        Endpoint,
        EndpointConfig,
        Model,
        ProductionVariant,
    )

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

        bucket = sagemaker_session.default_bucket()
        run_id = unique_name("mme")
        s3_key_prefix = f"tf220-inference-tests/mme-models/{run_id}"

        # Upload each tarball under the shared MME prefix so the runtime can
        # resolve target_model relative to the same S3 location.
        sagemaker_session.upload_data(path=model1_tar, bucket=bucket, key_prefix=s3_key_prefix)
        sagemaker_session.upload_data(path=model2_tar, bucket=bucket, key_prefix=s3_key_prefix)
        s3_model_prefix = f"s3://{bucket}/{s3_key_prefix}/"

        endpoint_name = unique_name("tf220-mme")
        model_name = unique_name("tf220-mme-model")
        cleanup_endpoint(endpoint_name, model_name=model_name)

        # 1. Create a multi-model SageMaker Model. The MME contract is
        #    expressed at the container definition level: mode="MultiModel"
        #    plus an S3 *prefix* (not a single tar) in model_data_url.
        Model.create(
            model_name=model_name,
            primary_container=ContainerDefinition(
                image=inference_image_uri,
                mode="MultiModel",
                model_data_url=s3_model_prefix,
            ),
            execution_role_arn=sagemaker_role_arn,
            session=boto_session,
        )

        # 2. Endpoint config + endpoint — same shape as single-model.
        EndpointConfig.create(
            endpoint_config_name=endpoint_name,
            production_variants=[
                ProductionVariant(
                    variant_name="AllTraffic",
                    model_name=model_name,
                    initial_instance_count=1,
                    instance_type=INSTANCE_TYPE,
                ),
            ],
            session=boto_session,
        )

        endpoint = Endpoint.create(
            endpoint_name=endpoint_name,
            endpoint_config_name=endpoint_name,
            session=boto_session,
        )
        endpoint.wait_for_status("InService")

        payload = json.dumps({"instances": [[1.0, 2.0, 3.0]]})

        # 3. Invoke each model by name. ``target_model`` maps to the
        #    X-Amzn-SageMaker-Target-Model header that selects the tarball
        #    within the MME's S3 prefix.
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
