"""Benchmark: training objective functions.

Migrated from SMFrameworksXGBoost3_0-5Tests/src/benchmarks/benchmark_training_objective.py
Tests: reg:squarederror, binary:logistic, multi:softmax (5/10/15 classes)
"""

import pytest

from .conftest import run_training_job

BASE_HP = {
    "max_depth": "5",
    "eta": "0.2",
    "gamma": "4",
    "min_child_weight": "6",
    "tree_method": "exact",
    "num_round": "50",
}


@pytest.mark.parametrize(
    "objective,dataset_path,extra_hp,timeout",
    [
        ("reg:squarederror", "xgboost/libsvm/100000x200", {}, 1200),
        ("binary:logistic", "xgboost/libsvm/binary", {}, 1200),
        ("multi:softmax", "xgboost/libsvm/multi/5", {"num_class": "5"}, 1800),
        ("multi:softmax", "xgboost/libsvm/multi/10", {"num_class": "10"}, 1800),
        ("multi:softmax", "xgboost/libsvm/multi/15", {"num_class": "15"}, 2400),
    ],
    ids=[
        "reg-squarederror-100kx200",
        "binary-logistic",
        "multi-softmax-5class",
        "multi-softmax-10class",
        "multi-softmax-15class",
    ],
)
def test_objective(image_uri, role, benchmark_bucket, objective, dataset_path, extra_hp, timeout):
    hp = {**BASE_HP, "objective": objective, **extra_hp}
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
