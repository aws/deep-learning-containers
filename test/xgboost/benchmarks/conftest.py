"""Benchmark test fixtures and helpers.

Replaces ai_algorithms_qa orchestration with direct SageMaker SDK calls.
All jobs run in the CI account using SageMakerRole (same as vLLM/SGLang tests).

Benchmark data must be available in the CI account's S3 bucket.
See test/xgboost/README.md for data setup instructions.
"""

import logging
import time

import boto3
import pytest
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from test_utils import random_suffix_name

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# Default bucket — override via --benchmark-bucket if data is in a different location
DEFAULT_BENCHMARK_BUCKET = "amazonai-algorithms-benchmarking"


def pytest_addoption(parser):
    parser.addoption(
        "--benchmark-bucket",
        action="store",
        default=DEFAULT_BENCHMARK_BUCKET,
        help="S3 bucket containing benchmark datasets",
    )


@pytest.fixture(scope="session")
def benchmark_bucket(request):
    return request.config.getoption("--benchmark-bucket")


def s3_uri(bucket, key):
    return f"s3://{bucket}/{key}"


def run_training_job(
    image_uri,
    role,
    hyperparameters,
    benchmark_bucket,
    train_s3_key,
    validation_s3_key,
    content_type,
    instance_type="ml.m5.large",
    instance_count=1,
    volume_size=5,
    max_run=1800,
    input_mode="File",
):
    """Launch a SageMaker training job and return (job_name, duration_seconds, job_description)."""
    job_name = random_suffix_name("xgb-bench", 50)
    output_path = s3_uri(benchmark_bucket, f"benchmark-output/{job_name}")

    estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        output_path=output_path,
        hyperparameters=hyperparameters,
        volume_size=volume_size,
        max_run=max_run,
        input_mode=input_mode,
    )

    channels = {
        "train": TrainingInput(
            s3_data=s3_uri(benchmark_bucket, train_s3_key), content_type=content_type
        ),
        "validation": TrainingInput(
            s3_data=s3_uri(benchmark_bucket, validation_s3_key),
            content_type=content_type,
        ),
    }

    LOGGER.info(f"Starting benchmark job: {job_name} ({instance_count}x {instance_type})")
    sm = boto3.client("sagemaker")
    start = time.time()
    try:
        estimator.fit(channels, job_name=job_name)
    except Exception:
        # Stop the training job if fit() fails (timeout, capacity, etc.)
        try:
            sm.stop_training_job(TrainingJobName=job_name)
        except Exception:
            pass
        raise
    duration = time.time() - start

    desc = sm.describe_training_job(TrainingJobName=job_name)
    LOGGER.info(
        f"Job {job_name} completed in {duration:.0f}s — status: {desc['TrainingJobStatus']}"
    )
    return job_name, duration, desc
