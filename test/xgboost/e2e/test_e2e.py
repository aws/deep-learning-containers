"""End-to-end training + inference tests.

Migrated from SMFrameworksXGBoost3_0-5Tests/src/integration_tests/test_e2e.py
Trains a model, then deploys it for real-time inference.
"""

import pytest

from .conftest import (
    data_uri,
    delete_endpoint,
    deploy_endpoint,
    run_training_job,
)

E2E_HP = {
    "max_depth": "2",
    "eta": "0.2",
    "gamma": "4",
    "min_child_weight": "6",
    "subsample": "0.7",
    "verbosity": "3",
    "objective": "reg:squarederror",
    "num_round": "10",
}


@pytest.fixture(scope="module")
def trained_model(image_uri, role):
    """Train a model once for all e2e tests in this module."""
    _, _, desc = run_training_job(
        image_uri=image_uri, role=role, hyperparameters=E2E_HP,
        train_s3_key="train", validation_s3_key="test",
        content_type="text/libsvm", test_name="e2e-train",
    )
    assert desc["TrainingJobStatus"] == "Completed"
    return desc["ModelArtifacts"]["S3ModelArtifacts"]


class TestE2E:
    def test_train_and_deploy(self, image_uri, role, trained_model):
        endpoint_name = None
        try:
            predictor, endpoint_name = deploy_endpoint(
                image_uri=image_uri, role=role,
                model_data=trained_model, test_name="e2e-infer",
            )
            predictor.content_type = "text/libsvm"
            predictor.accept = "text/csv"

            payload = "3 1:0.5 2:0.3 3:0.1 4:0.2 5:0.6 6:0.4 7:0.8 8:0.9"
            response = predictor.predict(payload)
            assert response is not None
            assert len(response) > 0
        finally:
            if endpoint_name:
                delete_endpoint(endpoint_name)

    def test_gpu_train(self, image_uri, role):
        hp = {**E2E_HP, "tree_method": "gpu_hist"}
        _, duration, desc = run_training_job(
            image_uri=image_uri, role=role, hyperparameters=hp,
            train_s3_key="train", validation_s3_key="test",
            content_type="text/libsvm", test_name="e2e-gpu",
            instance_type="ml.g4dn.2xlarge",
        )
        assert desc["TrainingJobStatus"] == "Completed"
