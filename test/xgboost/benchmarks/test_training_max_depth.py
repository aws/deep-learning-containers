"""Benchmark: max_depth parameter.

Migrated from SMFrameworksXGBoost3_0-5Tests/src/benchmarks/benchmark_training_max_depth.py
"""

import pytest

from .conftest import run_training_job

BASE_HP = {"eta": "0.2", "gamma": "4", "min_child_weight": "6", "tree_method": "exact"}


@pytest.mark.parametrize(
    "max_depth,dataset_path,extra_hp,timeout",
    [
        ("10", "xgboost/libsvm/mnist", {"objective": "multi:softmax", "num_class": "10", "num_round": "25"}, 1200),
        ("20", "xgboost/libsvm/mnist", {"objective": "multi:softmax", "num_class": "10", "num_round": "25"}, 1800),
        ("30", "xgboost/libsvm/mnist", {"objective": "multi:softmax", "num_class": "10", "num_round": "25"}, 1800),
        ("10", "xgboost/libsvm/100000x200", {"objective": "reg:squarederror", "num_round": "50"}, 1200),
        ("20", "xgboost/libsvm/100000x200", {"objective": "reg:squarederror", "num_round": "50"}, 1200),
        ("30", "xgboost/libsvm/100000x200", {"objective": "reg:squarederror", "num_round": "50"}, 2400),
    ],
    ids=["mnist-depth10", "mnist-depth20", "mnist-depth30", "synthetic-depth10", "synthetic-depth20", "synthetic-depth30"],
)
def test_max_depth(image_uri, role, benchmark_bucket, max_depth, dataset_path, extra_hp, timeout):
    hp = {**BASE_HP, "max_depth": max_depth, **extra_hp}
    _, duration, desc = run_training_job(
        image_uri=image_uri, role=role, benchmark_bucket=benchmark_bucket,
        hyperparameters=hp, train_s3_key=f"{dataset_path}/train/",
        validation_s3_key=f"{dataset_path}/val/", content_type="text/libsvm",
        instance_type="ml.m5.large", max_run=timeout,
    )
    assert desc["TrainingJobStatus"] == "Completed"
    assert 1 <= duration <= timeout
