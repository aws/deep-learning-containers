"""Single-model endpoint integration test for TF 2.20 inference DLC.

Builds a tiny ``y = 2x`` SavedModel, deploys it to a single-instance SageMaker
endpoint backed by the v2 inference image under test, and asserts the
predicted values.

Uses the SageMaker Python SDK v3 ``sagemaker-core`` resource layer
(``Endpoint``, ``EndpointConfig``, ``Model``, ``ContainerDefinition``,
``ProductionVariant``) — the v2 ``TensorFlowModel`` / ``Predictor`` classes
were removed in v3. ``ModelBuilder`` is the v3 entry point for
auto-detected deployments, but for DLC tests we already supply the
``image_uri`` and a pre-built ``model.tar.gz``, so we go straight to the
resource layer (the same surface ``ModelBuilder`` calls underneath).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from .resources.build_sample_model import build_sample_model

INSTANCE_TYPE = "ml.c5.xlarge"


def test_single_model_predict(
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

    with tempfile.TemporaryDirectory(prefix="tf220-single-") as workdir:
        tar_path = build_sample_model(
            output_dir=workdir,
            multiplier=2.0,
            model_name="model",
        )

        # Upload the tarball via the v3 helper Session — same default-bucket /
        # upload_data ergonomics as v2.
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

        # 1. Create the SageMaker Model — points at our DLC image and the
        #    uploaded SavedModel tar.gz.
        Model.create(
            model_name=model_name,
            primary_container=ContainerDefinition(
                image=inference_image_uri,
                model_data_url=model_data,
            ),
            execution_role_arn=sagemaker_role_arn,
            session=boto_session,
        )

        # 2. Create the EndpointConfig with a single ProductionVariant.
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

        # 3. Create the Endpoint and wait for it to come InService.
        endpoint = Endpoint.create(
            endpoint_name=endpoint_name,
            endpoint_config_name=endpoint_name,
            session=boto_session,
        )
        endpoint.wait_for_status("InService")

        # 4. Invoke. ``Endpoint.invoke`` returns an InvokeEndpointOutput whose
        #    ``body`` is a streaming bytes-like object.
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

        # Output signature is {"output": x * 2.0} -> TFS surfaces the tensor
        # under the signature output key when there is a single named tensor;
        # some TFS versions instead return the raw list. Handle both.
        first = predictions[0]
        if isinstance(first, dict) and "output" in first:
            values = first["output"]
        else:
            values = first

        assert values == pytest.approx([2.0, 4.0, 6.0]), f"got {values!r}"
