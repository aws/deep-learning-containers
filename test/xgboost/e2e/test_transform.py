"""Batch transform tests.

Migrated from SMFrameworksXGBoost3_0-5Tests/src/integration_tests/test_transform.py
"""

import pytest

from .conftest import data_uri, run_batch_transform, run_training_job

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
    """Train a model once for transform tests."""
    _, _, desc = run_training_job(
        image_uri=image_uri,
        role=role,
        hyperparameters=TRAIN_HP,
        train_s3_key="train",
        validation_s3_key="test",
        content_type="text/libsvm",
        test_name="bt-model",
    )
    assert desc["TrainingJobStatus"] == "Completed"
    return desc["ModelArtifacts"]["S3ModelArtifacts"]


class TestTransform:
    def test_batch_inference_libsvm(self, image_uri, role, model_data):
        desc = run_batch_transform(
            image_uri=image_uri,
            role=role,
            model_data=model_data,
            input_s3_uri=data_uri("test/abalone.test"),
            content_type="text/libsvm",
            test_name="bt-libsvm",
        )
        assert desc["TransformJobStatus"] == "Completed"
