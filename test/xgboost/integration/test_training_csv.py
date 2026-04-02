"""Training tests with CSV content type.

Migrated from SMFrameworksXGBoost3_0-5Tests/src/integration_tests/test_training_csv.py
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


class TestTrainingCsv:
    def test_single_instance(self, image_uri, role):
        _, duration, desc = run_training_job(
            image_uri=image_uri,
            role=role,
            hyperparameters=BASE_HP,
            train_s3_key="csv/train",
            validation_s3_key="csv/test",
            content_type="text/csv",
        )
        assert desc["TrainingJobStatus"] == "Completed"
        assert 1 <= duration <= 1800

    def test_distributed(self, image_uri, role):
        hp = {**BASE_HP, "tree_method": "hist"}
        _, duration, desc = run_training_job(
            image_uri=image_uri,
            role=role,
            hyperparameters=hp,
            train_s3_key="csv/train",
            validation_s3_key="csv/test",
            content_type="text/csv",
            instance_count=2,
        )
        assert desc["TrainingJobStatus"] == "Completed"

    def test_pipe_mode_single_instance(self, image_uri, role):
        _, duration, desc = run_training_job(
            image_uri=image_uri,
            role=role,
            hyperparameters=BASE_HP,
            train_s3_key="csv/train",
            validation_s3_key="csv/test",
            content_type="text/csv",
            input_mode="Pipe",
        )
        assert desc["TrainingJobStatus"] == "Completed"
