"""Inference tests — both single-model and multi-model endpoints against the
sklearn MME model tarballs. Verifies both endpoint modes round-trip a
CSV payload.
"""

import pytest

from .conftest import (
    data_uri,
    delete_endpoint,
    deploy_endpoint,
    deploy_multi_model_endpoint,
)

# MME lazy-loads only the tarball named in `target_model`, so the sibling
# `code/user_code.tar.gz` under this prefix is never invoked as a model.
MME_MODEL_PREFIX = "mme_models/"
MME_TARGET_MODELS = ["sklearn_1_model_0.tar.gz", "sklearn_1_model_1.tar.gz"]
MME_CODE_KEY = "mme_models/code/user_code.tar.gz"

# Fitted models have n_features_in_ = 6. Multi-row CSV exercises batch predict.
SAMPLE_PAYLOAD = "\n".join(
    [
        "0.0, 0.0, 0.0, 0.0, 0.0, 0.0",
        "0.0, 2.0, 4.0, 6.0, 9.0, 3.0",
    ]
)


@pytest.fixture(scope="module")
def mme_env():
    return {
        "SAGEMAKER_PROGRAM": "script.py",
        "SAGEMAKER_SUBMIT_DIRECTORY": data_uri(MME_CODE_KEY),
    }


class TestInferenceMME:
    def test_csv_single_model(self, image_uri, role, mme_env):
        endpoint_name = None
        try:
            predictor, endpoint_name = deploy_endpoint(
                image_uri=image_uri,
                role=role,
                model_data=data_uri(f"{MME_MODEL_PREFIX}{MME_TARGET_MODELS[0]}"),
                env=mme_env,
                test_name="inference-single",
            )
            predictor.content_type = "text/csv"
            predictor.accept = "text/csv"
            response = predictor.predict(SAMPLE_PAYLOAD)
            assert response is not None
        finally:
            if endpoint_name:
                delete_endpoint(endpoint_name)

    def test_csv_multimodel(self, image_uri, role, mme_env):
        endpoint_name = None
        try:
            predictor, endpoint_name, _ = deploy_multi_model_endpoint(
                image_uri=image_uri,
                role=role,
                model_data_prefix=data_uri(MME_MODEL_PREFIX),
                env=mme_env,
                test_name="inference-mme",
            )
            predictor.content_type = "text/csv"
            predictor.accept = "text/csv"
            for target_model in MME_TARGET_MODELS:
                response = predictor.predict(SAMPLE_PAYLOAD, target_model=target_model)
                assert response is not None
        finally:
            if endpoint_name:
                delete_endpoint(endpoint_name)
