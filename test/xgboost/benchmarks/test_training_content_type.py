"""Benchmark: content type / input mode.

Migrated from SMFrameworksXGBoost3_0-5Tests/src/benchmarks/benchmark_training_content_type.py
Note: Pipe mode removed for recordio-protobuf and parquet as XGBoost
algorithm mode does not reliably support pipe input for these formats.
"""

import pytest

from .conftest import run_training_job

BASE_HP = {
    "max_depth": "5",
    "eta": "0.2",
    "gamma": "4",
    "min_child_weight": "6",
    "objective": "reg:squarederror",
    "tree_method": "exact",
    "num_round": "50",
}


@pytest.mark.parametrize(
    "dataset_path,content_type,input_mode",
    [
        ("xgboost/libsvm/500000x1000", "text/libsvm", "File"),
        ("xgboost/csv/500000x1000", "text/csv", "File"),
        ("xgboost/csv/500000x1000", "text/csv", "Pipe"),
        (
            "xgboost/recordio-protobuf/500000x1000",
            "application/x-recordio-protobuf",
            "File",
        ),
        ("xgboost/parquet/500000x1000", "application/x-parquet", "File"),
    ],
    ids=[
        "libsvm-file",
        "csv-file",
        "csv-pipe",
        "recordio-protobuf-file",
        "parquet-file",
    ],
)
def test_content_type(
    image_uri, role, benchmark_bucket, dataset_path, content_type, input_mode
):
    _, duration, desc = run_training_job(
        image_uri=image_uri,
        role=role,
        benchmark_bucket=benchmark_bucket,
        hyperparameters=BASE_HP,
        train_s3_key=f"{dataset_path}/train/",
        validation_s3_key=f"{dataset_path}/val/",
        content_type=content_type,
        instance_type="ml.m5.2xlarge",
        volume_size=20,
        max_run=1800,
        input_mode=input_mode,
    )
    assert desc["TrainingJobStatus"] == "Completed"
    assert 1 <= duration <= 1800
