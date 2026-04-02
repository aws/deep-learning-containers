"""Inference tests — train a model first, then test inference formats.

Migrated from SMFrameworksXGBoost3_0-5Tests/src/integration_tests/test_inference.py
"""

import pytest

from .conftest import delete_endpoint, deploy_endpoint, run_training_job

TRAIN_HP = {
    "max_depth": "5",
    "eta": "0.2",
    "gamma": "4",
    "min_child_weight": "6",
    "subsample": "0.7",
    "verbosity": "3",
    "objective": "reg:squarederror",
    "num_round": "10",
}


@pytest.fixture(scope="module")
def model_data(image_uri, role):
    """Train a model once for all inference tests."""
    _, _, desc = run_training_job(
        image_uri=image_uri, role=role, hyperparameters=TRAIN_HP,
        train_s3_key="train", validation_s3_key="test",
        content_type="text/libsvm", test_name="infer-model",
    )
    assert desc["TrainingJobStatus"] == "Completed"
    return desc["ModelArtifacts"]["S3ModelArtifacts"]


class TestInference:
    def test_libsvm_inference(self, image_uri, role, model_data):
        endpoint_name = None
        try:
            predictor, endpoint_name = deploy_endpoint(
                image_uri=image_uri, role=role,
                model_data=model_data, test_name="infer-libsvm",
            )
            predictor.content_type = "text/libsvm"
            predictor.accept = "text/csv"

            payload = "3 1:0.5 2:0.3"
            response = predictor.predict(payload)
            assert response is not None
        finally:
            if endpoint_name:
                delete_endpoint(endpoint_name)

    def test_csv_inference(self, image_uri, role, model_data):
        endpoint_name = None
        try:
            predictor, endpoint_name = deploy_endpoint(
                image_uri=image_uri, role=role,
                model_data=model_data, test_name="infer-csv",
            )
            predictor.content_type = "text/csv"
            predictor.accept = "text/csv"

            payload = "0.5,0.3"
            response = predictor.predict(payload)
            assert response is not None
        finally:
            if endpoint_name:
                delete_endpoint(endpoint_name)
