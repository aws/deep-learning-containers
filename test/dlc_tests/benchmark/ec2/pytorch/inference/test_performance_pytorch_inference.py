import os
import time
import pytest
import logging
import sys

from src.benchmark_metrics import (
    PYTORCH_INFERENCE_GPU_THRESHOLD,
    PYTORCH_INFERENCE_CPU_THRESHOLD,
    get_threshold_for_image,
)
from test.test_utils import (
    CONTAINER_TESTS_PREFIX,
    get_framework_and_version_from_tag,
    UL22_BASE_ARM64_DLAMI_US_WEST_2,
    login_to_ecr_registry,
    get_account_id_from_image_uri,
    LOGGER,
)
from test.test_utils.ec2 import (
    ec2_performance_upload_result_to_s3_and_validate,
    post_process_inference,
    get_ec2_instance_type,
)


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stderr))


PT_PERFORMANCE_INFERENCE_SCRIPT = os.path.join(
    CONTAINER_TESTS_PREFIX, "benchmark", "run_pytorch_inference_performance.py"
)
PT_PERFORMANCE_INFERENCE_CPU_CMD = f"{PT_PERFORMANCE_INFERENCE_SCRIPT} --iterations 500 "
PT_PERFORMANCE_INFERENCE_GPU_CMD = f"{PT_PERFORMANCE_INFERENCE_SCRIPT} --iterations 1000 --gpu"
# Use the original p3.16xlarge instance, consider if use single gpu instance like g4dn.4xlarge, g5.4xlarge
PT_EC2_GPU_INSTANCE_TYPE = ["p3.16xlarge"]
PT_EC2_CPU_INSTANCE_TYPE = get_ec2_instance_type(default="c5.18xlarge", processor="cpu")
# c6g.4xlarge c6g.8xlarge reaches the 100% cpu usage for the benchmark when run VGG13 model
PT_EC2_CPU_ARM64_INSTANCE_TYPES = ["c7g.4xlarge", "c8g.4xlarge", "m7g.4xlarge", "r8g.4xlarge"]
PT_EC2_GPU_ARM64_INSTANCE_TYPE = get_ec2_instance_type(
    default="g5g.4xlarge", processor="gpu", arch_type="arm64"
)


@pytest.mark.model(
    "VGG13, MobileNet_V2, GoogLeNet, DenseNet121, Inception_V3, ResNet18, ResNet50, ViT_B_16, Bert_128, Bert_256, Roberta_128, Roberta_256, DistilBert_128, DistilBert_256, All-MPNet_128, All-MPNet_256, ASR"
)
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INSTANCE_TYPE, indirect=True)
@pytest.mark.team("conda")
def test_performance_ec2_pytorch_inference_gpu(
    pytorch_inference, ec2_connection, region, gpu_only, ec2_instance_type
):
    _, framework_version = get_framework_and_version_from_tag(pytorch_inference)
    threshold = get_threshold_for_image(framework_version, PYTORCH_INFERENCE_GPU_THRESHOLD)
    ec2_performance_pytorch_inference(
        pytorch_inference,
        "gpu",
        ec2_instance_type,
        ec2_connection,
        region,
        PT_PERFORMANCE_INFERENCE_GPU_CMD,
        threshold,
    )


@pytest.mark.model(
    "ResNet18, MobileNet_V2, GoogLeNet, DenseNet121, Inception_V3, Bert_128, Roberta_128, DistilBert_128, All-MPNet_128, ASR"
)
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_CPU_INSTANCE_TYPE, indirect=True)
@pytest.mark.team("conda")
def test_performance_ec2_pytorch_inference_cpu(
    pytorch_inference, ec2_connection, region, cpu_only, ec2_instance_type
):
    _, framework_version = get_framework_and_version_from_tag(pytorch_inference)
    threshold = get_threshold_for_image(framework_version, PYTORCH_INFERENCE_CPU_THRESHOLD)
    ec2_performance_pytorch_inference(
        pytorch_inference,
        "cpu",
        ec2_instance_type,
        ec2_connection,
        region,
        PT_PERFORMANCE_INFERENCE_CPU_CMD,
        threshold,
    )


@pytest.mark.model(
    "VGG13, MobileNet_V2, GoogLeNet, DenseNet121, Inception_V3, ResNet18, ResNet50, ViT_B_16, Bert_128, Bert_256, Roberta_128, Roberta_256, DistilBert_128, DistilBert_256, All-MPNet_128, All-MPNet_256, ASR"
)
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_ARM64_INSTANCE_TYPE, indirect=True)
@pytest.mark.parametrize("ec2_instance_ami", [UL22_BASE_ARM64_DLAMI_US_WEST_2], indirect=True)
@pytest.mark.team("conda")
def test_performance_ec2_pytorch_inference_arm64_gpu(
    pytorch_inference_arm64, ec2_connection, region, gpu_only, ec2_instance_type
):
    _, framework_version = get_framework_and_version_from_tag(pytorch_inference_arm64)
    threshold = get_threshold_for_image(framework_version, PYTORCH_INFERENCE_GPU_THRESHOLD)
    if "arm64" not in pytorch_inference_arm64:
        pytest.skip("skip benchmark tests for non-arm64 images")
    ec2_performance_pytorch_inference(
        pytorch_inference_arm64,
        "gpu",
        ec2_instance_type,
        ec2_connection,
        region,
        PT_PERFORMANCE_INFERENCE_GPU_CMD,
        threshold,
    )


@pytest.mark.model(
    "ResNet18, MobileNet_V2, GoogLeNet, DenseNet121, Inception_V3, Bert_128, Roberta_128, DistilBert_128, All-MPNet_128, ASR"
)
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_CPU_ARM64_INSTANCE_TYPES, indirect=True)
@pytest.mark.parametrize("ec2_instance_ami", [UL22_BASE_ARM64_DLAMI_US_WEST_2], indirect=True)
@pytest.mark.team("conda")
def test_performance_ec2_pytorch_inference_arm64_cpu(
    pytorch_inference_arm64, ec2_connection, region, cpu_only, ec2_instance_type
):
    _, framework_version = get_framework_and_version_from_tag(pytorch_inference_arm64)
    threshold = get_threshold_for_image(framework_version, PYTORCH_INFERENCE_CPU_THRESHOLD)
    if "arm64" not in pytorch_inference_arm64:
        pytest.skip("skip benchmark tests for non-arm64 images")

    ec2_performance_pytorch_inference(
        pytorch_inference_arm64,
        "cpu",
        ec2_instance_type,
        ec2_connection,
        region,
        PT_PERFORMANCE_INFERENCE_CPU_CMD,
        threshold,
    )


def ec2_performance_pytorch_inference(
    image_uri, processor, ec2_instance_type, ec2_connection, region, test_cmd, threshold
):
    docker_runtime = "--runtime=nvidia --gpus all" if processor == "gpu" else ""
    container_test_local_dir = os.path.join("$HOME", "container_tests")
    repo_name, image_tag = image_uri.split("/")[-1].split(":")

    # Make sure we are logged into ECR so we can pull the image
    account_id = get_account_id_from_image_uri(image_uri)
    login_to_ecr_registry(ec2_connection, account_id, region)

    ec2_connection.run(f"docker pull -q {image_uri} ")

    time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
    commit_info = os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION")
    # Run performance inference command, display benchmark results to console

    container_name = f"{repo_name}-performance-{image_tag}-ec2"
    log_file = f"synthetic_{commit_info}_{time_str}.log"

    try:
        ec2_connection.run(
            f"docker run {docker_runtime} -d --name {container_name} -e OMP_NUM_THREADS=1 "
            f"-v {container_test_local_dir}:{os.path.join(os.sep, 'test')} {image_uri}"
        )

    except Exception as e:
        LOGGER.info(f"Failed to start container: {e}")
        return

    try:
        ec2_connection.run(f"docker exec {container_name} pip install transformers", warn=True)

    except Exception as e:
        LOGGER.info(f"Failed to install transformers: {e}")
        return

    try:
        LOGGER.info(f"Starting benchmark test on {processor} {ec2_instance_type} instance...")
        result = ec2_connection.run(
            f"docker exec {container_name} python {test_cmd} --instance {ec2_instance_type} 2>&1 | tee {log_file}",
            timeout=3600,
            warn=True,
        )

        # Check if the command was successful
        if result.failed:
            LOGGER.info(f"Command failed with exit code {result.return_code}")
            LOGGER.info(f"Error output:\n{result.stderr}")
        else:
            LOGGER.info("Command completed successfully")
        sys.stdout.flush()

    except Exception as e:
        LOGGER.info(f"An error occurred during test execution: {e}")

    finally:
        # This block will run regardless of whether an exception occurred
        LOGGER.info(f"Cleaning {processor} {ec2_instance_type} up...")

        ec2_connection.run(f"docker rm -f {container_name}")

    ec2_performance_upload_result_to_s3_and_validate(
        ec2_connection,
        image_uri,
        log_file,
        "synthetic",
        threshold,
        post_process_inference,
        log_file,
        ec2_instance_type,
    )
