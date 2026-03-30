"""Benchmark: tree methods (approx, exact, hist).

Migrated from SMFrameworksXGBoost3_0-5Tests/src/benchmarks/benchmark_training_tree_method.py
"""

import pytest

from .conftest import run_training_job

BASE_HP = {"eta": "0.2", "gamma": "4", "min_child_weight": "6"}


@pytest.mark.parametrize(
    "tree_method,dataset_path,extra_hp,timeout",
    [
        ("approx", "xgboost/libsvm/mnist", {"max_depth": "5", "objective": "multi:softmax", "num_class": "10", "num_round": "25"}, 1800),
        ("exact", "xgboost/libsvm/mnist", {"max_depth": "5", "objective": "multi:softmax", "num_class": "10", "num_round": "25"}, 1200),
        ("hist", "xgboost/libsvm/mnist", {"max_depth": "5", "objective": "multi:softmax", "num_class": "10", "num_round": "25"}, 1200),
        ("approx", "xgboost/libsvm/100000x200", {"max_depth": "10", "objective": "reg:squarederror", "num_round": "50"}, 1200),
        ("exact", "xgboost/libsvm/100000x200", {"max_depth": "10", "objective": "reg:squarederror", "num_round": "50"}, 1200),
        ("hist", "xgboost/libsvm/100000x200", {"max_depth": "10", "objective": "reg:squarederror", "num_round": "50"}, 1200),
    ],
    ids=["mnist-approx", "mnist-exact", "mnist-hist", "synthetic-approx", "synthetic-exact", "synthetic-hist"],
)
def test_tree_method(image_uri, role, benchmark_bucket, tree_method, dataset_path, extra_hp, timeout):
    hp = {**BASE_HP, "tree_method": tree_method, **extra_hp}
    _, duration, desc = run_training_job(
        image_uri=image_uri, role=role, benchmark_bucket=benchmark_bucket,
        hyperparameters=hp, train_s3_key=f"{dataset_path}/train/",
        validation_s3_key=f"{dataset_path}/val/", content_type="text/libsvm",
        instance_type="ml.m5.large", max_run=timeout,
    )
    assert desc["TrainingJobStatus"] == "Completed"
    assert 1 <= duration <= timeout
