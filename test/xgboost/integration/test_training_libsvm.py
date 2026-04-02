"""Training tests with libsvm content type.

Migrated from SMFrameworksXGBoost3_0-5Tests/src/integration_tests/test_training_libsvm.py
"""

import pytest

from .conftest import run_training_job

BASE_HP = {
    "max_depth": "5",
    "eta": "0.2",
    "gamma": "4",
    "min_child_weight": "6",
    "subsample": "0.7",
    "verbosity": "3",
    "objective": "reg:squarederror",
    "num_round": "10",
}


class TestTrainingLibsvm:
    def test_single_instance(self, image_uri, role):
        _, duration, desc = run_training_job(
            image_uri=image_uri,
            role=role,
            hyperparameters=BASE_HP,
            train_s3_key="train",
            validation_s3_key="test",
            content_type="text/libsvm",
        )
        assert desc["TrainingJobStatus"] == "Completed"
        assert 1 <= duration <= 1800

    def test_distributed(self, image_uri, role):
        hp = {**BASE_HP, "tree_method": "hist"}
        hp.pop("updater", None)
        _, duration, desc = run_training_job(
            image_uri=image_uri,
            role=role,
            hyperparameters=hp,
            train_s3_key="train",
            validation_s3_key="test",
            content_type="text/libsvm",
            instance_count=2,
        )
        assert desc["TrainingJobStatus"] == "Completed"

    def test_checkpoint_single_instance(self, image_uri, role):
        checkpoint_uri = f"s3://amazonai-algorithms-integration-tests/integ-output/checkpoints/{__name__}"
        _, duration, desc = run_training_job(
            image_uri=image_uri,
            role=role,
            hyperparameters=BASE_HP,
            train_s3_key="train",
            validation_s3_key="test",
            content_type="text/libsvm",
            checkpoint_s3_uri=checkpoint_uri,
        )
        assert desc["TrainingJobStatus"] == "Completed"

    def test_gpu_single_instance(self, image_uri, role):
        hp = {**BASE_HP, "tree_method": "gpu_hist"}
        _, duration, desc = run_training_job(
            image_uri=image_uri,
            role=role,
            hyperparameters=hp,
            train_s3_key="train",
            validation_s3_key="test",
            content_type="text/libsvm",
            instance_type="ml.g4dn.2xlarge",
        )
        assert desc["TrainingJobStatus"] == "Completed"
