import os

import pytest
import re

from test.test_utils import CONTAINER_TESTS_PREFIX, PT_GPU_PY3_BENCHMARK_IMAGENET_AMI_US_WEST_2, DEFAULT_REGION, \
    BENCHMARK_RESULTS_S3_BUCKET, LOGGER
from test.test_utils.ec2 import execute_ec2_training_performance_test
from src.benchmark_metrics import PYTORCH_TRAINING_GPU_SYNTHETIC_THRESHOLD, PYTORCH_TRAINING_GPU_IMAGENET_THRESHOLD

PT_PERFORMANCE_TRAINING_GPU_SYNTHETIC_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "benchmark",
                                                         "run_pytorch_training_performance_gpu_synthetic")
PT_PERFORMANCE_TRAINING_GPU_IMAGENET_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "benchmark",
                                                        "run_pytorch_training_performance_gpu_imagenet")

PT_EC2_GPU_SYNTHETIC_INSTANCE_TYPE = "p3.16xlarge"
PT_EC2_GPU_IMAGENET_INSTANCE_TYPE = "p3.16xlarge"


@pytest.mark.model("resnet50")
@pytest.mark.parametrize("ec2_instance_type", [PT_EC2_GPU_SYNTHETIC_INSTANCE_TYPE], indirect=True)
def test_performance_pytorch_gpu_synthetic(pytorch_training, ec2_connection, gpu_only, py3_only):
    execute_ec2_training_performance_test(ec2_connection, pytorch_training, PT_PERFORMANCE_TRAINING_GPU_SYNTHETIC_CMD,
                                          post_process=post_process_pytorch_gpu_py3_synthetic_ec2_training_performance_test,
                                          log_name_prefix="synthetic_results")

@pytest.mark.skip()
@pytest.mark.model("resnet50")
@pytest.mark.parametrize("ec2_instance_ami", [PT_GPU_PY3_BENCHMARK_IMAGENET_AMI_US_WEST_2], indirect=True)
@pytest.mark.parametrize("ec2_instance_type", [PT_EC2_GPU_IMAGENET_INSTANCE_TYPE], indirect=True)
def test_performance_pytorch_gpu_imagenet(pytorch_training, ec2_connection, gpu_only, py3_only):
    execute_pytorch_gpu_py3_imagenet_ec2_training_performance_test(ec2_connection, pytorch_training,
                                                                   PT_PERFORMANCE_TRAINING_GPU_IMAGENET_CMD)


def execute_pytorch_gpu_py3_imagenet_ec2_training_performance_test(connection, ecr_uri, test_cmd,
                                                                   region=DEFAULT_REGION):
    repo_name, image_tag = ecr_uri.split("/")[-1].split(":")
    container_test_local_dir = os.path.join("$HOME", "container_tests")

    container_name = f"{repo_name}-performance-{image_tag}-ec2"

    # Make sure we are logged into ECR so we can pull the image
    connection.run(f"$(aws ecr get-login --no-include-email --region {region})", hide=True)

    connection.run(f"nvidia-docker pull -q {ecr_uri}")

    # Run training command, display benchmark results to console
    try:
        connection.run(
            f"nvidia-docker run --user root -e COMMIT_INFO={os.getenv('CODEBUILD_RESOLVED_SOURCE_VERSION')} --shm-size 8G --env OMP_NUM_THREADS=1 --name {container_name} "
            f"-v {container_test_local_dir}:{os.path.join(os.sep, 'test')} "
            f"-v /home/ubuntu/:/root/:delegated "
            f"{ecr_uri} {os.path.join(os.sep, 'bin', 'bash')} -c {test_cmd}")
    finally:
        connection.run(f"docker rm -f {container_name}", warn=True, hide=True)


def post_process_pytorch_gpu_py3_synthetic_ec2_training_performance_test(connection, ecr_uri, log_location, log_name):
    framework_version = re.search(r"[0-9]+(\.\d+){2}", ecr_uri).group()
    py_version = "py2" if "py2" in ecr_uri else "py37" if "py37" in ecr_uri else "py3"

    s3_location = os.path.join(
        BENCHMARK_RESULTS_S3_BUCKET, "pytorch", framework_version, "ec2", "training", "gpu", py_version, log_name
    )
    LOGGER.info(f"Benchmark Results:\n"
                f"PyTorch {framework_version} Training gpu {py_version}")

    connection.run(f"tail {log_location} >&2")
    connection.run(
        f"aws s3 cp {log_location} {s3_location}")
    connection.run(
        f"echo To retrieve complete benchmark log, check {s3_location} >&2")

