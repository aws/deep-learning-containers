"""Benchmark: num_round parameter.

Migrated from SMFrameworksXGBoost3_0-5Tests/src/benchmarks/benchmark_training_num_round.py
"""

import pytest

from .conftest import run_training_job

BASE_HP = {"eta": "0.2", "gamma": "4", "min_child_weight": "6", "tree_method": "exact"}


@pytest.mark.parametrize(
    "num_round,dataset_path,extra_hp,timeout",
    [
        (
            "25",
            "xgboost/libsvm/mnist",
            {"max_depth": "5", "objective": "multi:softmax", "num_class": "10"},
            1200,
        ),
        (
            "50",
            "xgboost/libsvm/mnist",
            {"max_depth": "5", "objective": "multi:softmax", "num_class": "10"},
            1800,
        ),
        (
            "100",
            "xgboost/libsvm/mnist",
            {"max_depth": "5", "objective": "multi:softmax", "num_class": "10"},
            1800,
        ),
        (
            "50",
            "xgboost/libsvm/100000x200",
            {"max_depth": "15", "objective": "reg:squarederror"},
            1200,
        ),
        (
            "100",
            "xgboost/libsvm/100000x200",
            {"max_depth": "15", "objective": "reg:squarederror"},
            1200,
        ),
        (
            "200",
            "xgboost/libsvm/100000x200",
            {"max_depth": "15", "objective": "reg:squarederror"},
            2400,
        ),
    ],
    ids=[
        "mnist-25rounds",
        "mnist-50rounds",
        "mnist-100rounds",
        "synthetic-50rounds",
        "synthetic-100rounds",
        "synthetic-200rounds",
    ],
)
def test_num_round(image_uri, role, benchmark_bucket, num_round, dataset_path, extra_hp, timeout):
    hp = {**BASE_HP, "num_round": num_round, **extra_hp}
    _, duration, desc = run_training_job(
        image_uri=image_uri,
        role=role,
        benchmark_bucket=benchmark_bucket,
        hyperparameters=hp,
        train_s3_key=f"{dataset_path}/train/",
        validation_s3_key=f"{dataset_path}/val/",
        content_type="text/libsvm",
        instance_type="ml.m5.large",
        max_run=timeout,
    )
    assert desc["TrainingJobStatus"] == "Completed"
    assert 1 <= duration <= timeout
