import os
import time
import pytest
from src.benchmark_metrics import (
    PYTORCH_INFERENCE_GPU_THRESHOLD,
    PYTORCH_INFERENCE_CPU_THRESHOLD,
    get_threshold_for_image,
)
from test.test_utils import CONTAINER_TESTS_PREFIX, get_framework_and_version_from_tag, AML2_CPU_ARM64_US_WEST_2, LOGGER
from test.test_utils.ec2 import (
    ec2_performance_upload_result_to_s3_and_validate,
    post_process_inference,
)

PT_PERFORMANCE_INFERENCE_SCRIPT = os.path.join(
    CONTAINER_TESTS_PREFIX, "benchmark", "run_pytorch_inference_performance.py"
)
PT_PERFORMANCE_INFERENCE_CPU_CMD = f"{PT_PERFORMANCE_INFERENCE_SCRIPT} --iterations 500"
PT_PERFORMANCE_INFERENCE_GPU_CMD = f"{PT_PERFORMANCE_INFERENCE_SCRIPT} --iterations 1000 --gpu"


@pytest.mark.model("resnet18, VGG13, MobileNetV2, GoogleNet, DenseNet121, InceptionV3")
@pytest.mark.parametrize("ec2_instance_type", ["p3.16xlarge"], indirect=True)
def test_performance_ec2_pytorch_inference_gpu(pytorch_inference, ec2_connection, region, gpu_only):
    _, framework_version = get_framework_and_version_from_tag(pytorch_inference)
    threshold = get_threshold_for_image(framework_version, PYTORCH_INFERENCE_GPU_THRESHOLD)
    ec2_performance_pytorch_inference(
        pytorch_inference, "gpu", ec2_connection, region, PT_PERFORMANCE_INFERENCE_GPU_CMD, threshold,
    )


@pytest.mark.model("resnet18, VGG13, MobileNetV2, GoogleNet, DenseNet121, InceptionV3")
@pytest.mark.parametrize("ec2_instance_type", ["c5.18xlarge"], indirect=True)
def test_performance_ec2_pytorch_inference_cpu(pytorch_inference, ec2_connection, region, cpu_only):
    _, framework_version = get_framework_and_version_from_tag(pytorch_inference)
    threshold = get_threshold_for_image(framework_version, PYTORCH_INFERENCE_CPU_THRESHOLD)
    ec2_performance_pytorch_inference(
        pytorch_inference, "cpu", ec2_connection, region, PT_PERFORMANCE_INFERENCE_CPU_CMD, threshold,
    )


@pytest.mark.model("resnet18, VGG13, MobileNetV2, GoogleNet, DenseNet121, InceptionV3")
@pytest.mark.parametrize("ec2_instance_type", ["c6g.4xlarge"], indirect=True)
@pytest.mark.parametrize("ec2_instance_ami", [AML2_CPU_ARM64_US_WEST_2], indirect=True)
def test_performance_ec2_pytorch_inference_graviton_cpu(pytorch_inference_graviton, ec2_connection, region, cpu_only):
    _, framework_version = get_framework_and_version_from_tag(pytorch_inference_graviton)
    threshold = get_threshold_for_image(framework_version, PYTORCH_INFERENCE_CPU_THRESHOLD)
    ec2_performance_pytorch_inference(
        pytorch_inference_graviton, "cpu", ec2_connection, region, PT_PERFORMANCE_INFERENCE_CPU_CMD, threshold,
    )


def ec2_performance_pytorch_inference(image_uri, processor, ec2_connection, region, test_cmd, threshold):
    docker_cmd = "nvidia-docker" if processor == "gpu" else "docker"
    container_test_local_dir = os.path.join("$HOME", "container_tests")
    repo_name, image_tag = image_uri.split("/")[-1].split(":")

    # Make sure we are logged into ECR so we can pull the image
    ec2_connection.run(f"$(aws ecr get-login --no-include-email --region {region})", hide=True)

    ec2_connection.run(f"{docker_cmd} pull -q {image_uri} ")

    time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
    commit_info = os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION")
    # Run performance inference command, display benchmark results to console
    container_name = f"{repo_name}-performance-{image_tag}-ec2"
    log_file = f"synthetic_{commit_info}_{time_str}.log"
    LOGGER.info(f"DEBUG: Running container {container_name}")
    ec2_connection.run(
        f"{docker_cmd} run -d --name {container_name}  -e OMP_NUM_THREADS=1 "
        f"-v {container_test_local_dir}:{os.path.join(os.sep, 'test')} {image_uri} "
    )
    LOGGER.info(f"DEBUG: Running test command {test_cmd}")
    ec2_connection.run(f"{docker_cmd} exec {container_name} " f"python {test_cmd} " f"2>&1 | tee {log_file}")
    LOGGER.info(f"DEBUG: Deleting container {container_name}")
    ec2_connection.run(f"docker rm -f {container_name}")
    LOGGER.info(f"DEBUG: Pushing results to s3 for log file {log_file}")
    ec2_performance_upload_result_to_s3_and_validate(
        ec2_connection, image_uri, log_file, "synthetic", threshold, post_process_inference, log_file,
    )
