"""Benchmark: data size scaling.

Migrated from SMFrameworksXGBoost3_0-5Tests/src/benchmarks/benchmark_training_data_size.py
"""

import pytest

from .conftest import run_training_job

BASE_HP = {
    "max_depth": "5", "eta": "0.2", "gamma": "4", "min_child_weight": "6",
    "objective": "reg:squarederror", "tree_method": "exact", "num_round": "50",
}


@pytest.mark.parametrize(
    "dataset_path,volume_size,timeout",
    [
        ("xgboost/libsvm/100000x200", 5, 1200),
        ("xgboost/libsvm/500000x200", 5, 1200),
        ("xgboost/libsvm/100000x1000", 5, 1200),
        ("xgboost/libsvm/500000x1000", 20, 1800),
    ],
    ids=["100kx200", "500kx200", "100kx1000", "500kx1000"],
)
def test_data_size(image_uri, role, benchmark_bucket, dataset_path, volume_size, timeout):
    _, duration, desc = run_training_job(
        image_uri=image_uri, role=role, benchmark_bucket=benchmark_bucket,
        hyperparameters=BASE_HP, train_s3_key=f"{dataset_path}/train/",
        validation_s3_key=f"{dataset_path}/val/", content_type="text/libsvm",
        instance_type="ml.m5.2xlarge", volume_size=volume_size, max_run=timeout,
    )
    assert desc["TrainingJobStatus"] == "Completed"
    assert 1 <= duration <= timeout
