import os
import time
import pytest
import re

from test.test_utils import (
    CONTAINER_TESTS_PREFIX,
    PT_GPU_PY3_BENCHMARK_IMAGENET_AMI_US_WEST_2,
    DEFAULT_REGION,
    get_framework_and_version_from_tag,
    is_pr_context,
)
from test.test_utils.ec2 import (
    execute_ec2_training_performance_test,
    ec2_performance_upload_result_to_s3_and_validate,
)
from src.benchmark_metrics import (
    PYTORCH_TRAINING_GPU_SYNTHETIC_THRESHOLD,
    PYTORCH_TRAINING_GPU_IMAGENET_THRESHOLD,
    get_threshold_for_image,
)

PT_PERFORMANCE_TRAINING_GPU_SYNTHETIC_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX, "benchmark", "run_pytorch_training_performance_gpu_synthetic",
)
PT_PERFORMANCE_TRAINING_GPU_IMAGENET_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX, "benchmark", "run_pytorch_training_performance_gpu_imagenet"
)

PT_EC2_GPU_SYNTHETIC_INSTANCE_TYPE = "p3.16xlarge"
PT_EC2_GPU_IMAGENET_INSTANCE_TYPE = "p3.16xlarge"


@pytest.mark.model("resnet50")
@pytest.mark.parametrize("ec2_instance_type", [PT_EC2_GPU_SYNTHETIC_INSTANCE_TYPE], indirect=True)
def test_performance_pytorch_gpu_synthetic(pytorch_training, ec2_connection, gpu_only, py3_only):
    _, framework_version = get_framework_and_version_from_tag(pytorch_training)
    threshold = get_threshold_for_image(framework_version, PYTORCH_TRAINING_GPU_SYNTHETIC_THRESHOLD)
    execute_ec2_training_performance_test(
        ec2_connection,
        pytorch_training,
        PT_PERFORMANCE_TRAINING_GPU_SYNTHETIC_CMD,
        post_process=post_process_pytorch_gpu_py3_synthetic_ec2_training_performance,
        data_source="synthetic",
        threshold={"Throughput": threshold},
    )


@pytest.mark.skip(reason="Current infrastructure issues are causing this to timeout.")
@pytest.mark.model("resnet50")
@pytest.mark.parametrize("ec2_instance_ami", [PT_GPU_PY3_BENCHMARK_IMAGENET_AMI_US_WEST_2], indirect=True)
@pytest.mark.parametrize("ec2_instance_type", [PT_EC2_GPU_IMAGENET_INSTANCE_TYPE], indirect=True)
def test_performance_pytorch_gpu_imagenet(pytorch_training, ec2_connection, gpu_only, py3_only):
    execute_pytorch_gpu_py3_imagenet_ec2_training_performance_test(
        ec2_connection, pytorch_training, PT_PERFORMANCE_TRAINING_GPU_IMAGENET_CMD
    )


def execute_pytorch_gpu_py3_imagenet_ec2_training_performance_test(
    connection, ecr_uri, test_cmd, region=DEFAULT_REGION
):
    _, framework_version = get_framework_and_version_from_tag(ecr_uri)
    threshold = get_threshold_for_image(framework_version, PYTORCH_TRAINING_GPU_IMAGENET_THRESHOLD)
    repo_name, image_tag = ecr_uri.split("/")[-1].split(":")
    container_test_local_dir = os.path.join("$HOME", "container_tests")

    container_name = f"{repo_name}-performance-{image_tag}-ec2"

    # Make sure we are logged into ECR so we can pull the image
    connection.run(f"$(aws ecr get-login --no-include-email --region {region})", hide=True)
    connection.run(f"nvidia-docker pull -q {ecr_uri}")
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    log_name = f"imagenet_{os.getenv('CODEBUILD_RESOLVED_SOURCE_VERSION')}_{timestamp}.txt"
    log_location = os.path.join(container_test_local_dir, "benchmark", "logs", log_name)
    # Run training command, display benchmark results to console
    try:
        connection.run(
            f"nvidia-docker run --user root "
            f"-e LOG_FILE={os.path.join(os.sep, 'test', 'benchmark', 'logs', log_name)} "
            f"-e PR_CONTEXT={1 if is_pr_context() else 0} "
            f"--shm-size 8G --env OMP_NUM_THREADS=1 --name {container_name} "
            f"-v {container_test_local_dir}:{os.path.join(os.sep, 'test')} "
            f"-v /home/ubuntu/:/root/:delegated "
            f"{ecr_uri} {os.path.join(os.sep, 'bin', 'bash')} -c {test_cmd}"
        )
    finally:
        connection.run(f"docker rm -f {container_name}", warn=True, hide=True)
    ec2_performance_upload_result_to_s3_and_validate(
        connection,
        ecr_uri,
        log_location,
        "imagenet",
        {"Cost": threshold},
        post_process_pytorch_gpu_py3_imagenet_ec2_training_performance,
        log_name,
    )


def post_process_pytorch_gpu_py3_synthetic_ec2_training_performance(connection, log_location):
    last_lines = connection.run(f"tail {log_location}").stdout.split("\n")
    throughput = 0
    for line in reversed(last_lines):
        if "__results.throughput__" in line:
            throughput = float(line.split("=")[1])
            break
    return {"Throughput": throughput}


def post_process_pytorch_gpu_py3_imagenet_ec2_training_performance(connection, log_location):
    log_content = connection.run(f"cat {log_location}").stdout.split("\n")
    cost = None
    for line in reversed(log_content):
        if "took time" in line:
            cost = float(re.search(r"(took time:[ ]*)(?P<cost>[0-9]+\.?[0-9]+)", line).group("cost"))
            break
    return {"Cost": cost}
