"""SageMaker training on Blackwell (G7, sm_120) for CUDA images built with sm_120.

Run only by the non-blocking sagemaker-test-blackwell job
(tensorflow-training.tests-sagemaker.yml), which sets SM_BLACKWELL=1. SageMaker
Training places ml.g7.2xlarge jobs on the CI account's G7 capacity reservation
once that ODCR is onboarded for SageMaker Training (shared with the training
compute accounts and allowlisted by the Training team), and on the on-demand
pool otherwise; ml.g7.12xlarge always comes from the on-demand pool. A capacity
shortage must not block a PR.

  - single-gpu: ml.g7.2xlarge, 1 x RTX PRO 4500 Blackwell
  - mirrored:   ml.g7.12xlarge, 2 x RTX PRO 4500 Blackwell, MirroredStrategy

Each job's CloudWatch log must show TensorFlow created every expected GPU at
compute capability 12.0: TensorFlow falls back to CPU silently when it cannot
use the GPU, and the job would still succeed. Jobs run with
CUDA_DISABLE_PTX_JIT=1, so a kernel without native sm_120 code fails instead
of being JIT-compiled from PTX.
"""

import os
import time

import boto3
import pytest
from sagemaker.core.training.configs import InputData

pytestmark = pytest.mark.skipif(
    os.environ.get("SM_BLACKWELL") != "1",
    reason="Blackwell training job only (SM_BLACKWELL=1, set by sagemaker-test-blackwell)",
)

BLACKWELL_CAPABILITY = "compute capability: 12.0"


def _job_log_lines(region, job_name, pattern, timeout=300):
    """Return /aws/sagemaker/TrainingJobs log lines for job_name matching pattern."""
    logs = boto3.client("logs", region_name=region)
    deadline = time.time() + timeout
    while True:
        lines = []
        try:
            paginator = logs.get_paginator("filter_log_events")
            for page in paginator.paginate(
                logGroupName="/aws/sagemaker/TrainingJobs",
                logStreamNamePrefix=f"{job_name}/",
                filterPattern=f'"{pattern}"',
            ):
                lines.extend(e["message"] for e in page.get("events", []))
        except logs.exceptions.ResourceNotFoundException:
            pass
        if lines or time.time() >= deadline:
            return lines
        time.sleep(15)


@pytest.mark.parametrize(
    "instance_type,strategy,gpus",
    [
        ("ml.g7.2xlarge", "none", 1),
        ("ml.g7.12xlarge", "mirrored", 2),
    ],
    ids=["single-gpu", "mirrored"],
)
def test_mnist_blackwell(
    instance_type, strategy, gpus, region, sm_trainer, assert_s3_file_exists, mnist_s3_uri
):
    model_trainer = sm_trainer(
        entry_script="mnist.py",
        instance_type=instance_type,
        instance_count=1,
        hyperparameters={"strategy": strategy},
        environment={"CUDA_DISABLE_PTX_JIT": "1"},
        input_data=[InputData(channel_name="training", data_source=mnist_s3_uri)],
        job_name_prefix=f"tf-mnist-g7-{strategy}",
    )
    job = model_trainer._latest_training_job
    assert_s3_file_exists(job.model_artifacts.s3_model_artifacts)

    lines = _job_log_lines(region, job.training_job_name, "Created device")
    print("\n".join(line.strip() for line in lines))
    for index in range(gpus):
        device = f"device:GPU:{index}"
        matching = [line for line in lines if device in line]
        assert matching, (
            f"{job.training_job_name}: no '{device}' in the job log; TensorFlow did not "
            f"register GPU {index} on {instance_type}"
        )
        assert any(BLACKWELL_CAPABILITY in line for line in matching), (
            f"{job.training_job_name}: {device} is not {BLACKWELL_CAPABILITY}: {matching}"
        )
