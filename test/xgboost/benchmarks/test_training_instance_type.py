"""Benchmark: instance type scaling (single + distributed).

Migrated from SMFrameworksXGBoost3_0-5Tests/src/benchmarks/benchmark_training_instance_type.py
"""

import pytest

from .conftest import run_training_job

BASE_HP = {
    "max_depth": "5",
    "eta": "0.2",
    "gamma": "4",
    "min_child_weight": "6",
    "objective": "reg:squarederror",
    "tree_method": "approx",
    "num_round": "100",
}


@pytest.mark.parametrize(
    "instance_type,instance_count,timeout",
    [
        ("ml.m5.large", 1, 1800),
        ("ml.m5.xlarge", 1, 1800),
        ("ml.m5.2xlarge", 1, 1800),
        ("ml.m5.large", 2, 1200),
        ("ml.m5.xlarge", 2, 1200),
        ("ml.m5.2xlarge", 2, 1200),
    ],
    ids=[
        "single-m5.large",
        "single-m5.xlarge",
        "single-m5.2xlarge",
        "distributed-m5.large",
        "distributed-m5.xlarge",
        "distributed-m5.2xlarge",
    ],
)
def test_instance_type(image_uri, role, benchmark_bucket, instance_type, instance_count, timeout):
    _, duration, desc = run_training_job(
        image_uri=image_uri,
        role=role,
        benchmark_bucket=benchmark_bucket,
        hyperparameters=BASE_HP,
        train_s3_key="xgboost/libsvm/500000x200/train/",
        validation_s3_key="xgboost/libsvm/500000x200/val/",
        content_type="text/libsvm",
        instance_type=instance_type,
        instance_count=instance_count,
        max_run=timeout,
    )
    assert desc["TrainingJobStatus"] == "Completed"
    assert 1 <= duration <= timeout
