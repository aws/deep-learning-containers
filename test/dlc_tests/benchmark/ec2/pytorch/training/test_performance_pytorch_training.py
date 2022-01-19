import os
import time
import pytest
import re
import statistics

from test.test_utils import (
    CONTAINER_TESTS_PREFIX,
    PT_GPU_PY3_BENCHMARK_IMAGENET_AMI_US_WEST_2,
    HPU_AL2_DLAMI,
    DEFAULT_REGION,
    get_framework_and_version_from_tag,
    get_synapseai_version_from_tag,
    is_pr_context,
)
from test.test_utils.ec2 import (
    execute_ec2_training_performance_test,
    ec2_performance_upload_result_to_s3_and_validate,
)
from src.benchmark_metrics import (
    PYTORCH_TRAINING_RN50_HPU_SYNTHETIC_1_CARD_THRESHOLD,
    PYTORCH_TRAINING_RN50_HPU_SYNTHETIC_8_CARD_THRESHOLD,
    PYTORCH_TRAINING_BERT_HPU_THRESHOLD,
    PYTORCH_TRAINING_GPU_SYNTHETIC_THRESHOLD,
    PYTORCH_TRAINING_GPU_IMAGENET_THRESHOLD,
    get_threshold_for_image,
)

PT_PERFORMANCE_RN50_TRAINING_HPU_SYNTHETIC_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX, "benchmark", "run_pytorch_rn50_training_performance_hpu_synthetic",
)
PT_PERFORMANCE_BERT_TRAINING_HPU_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX, "benchmark", "run_pytorch_bert_training_performance_hpu",
)
PT_PERFORMANCE_TRAINING_GPU_SYNTHETIC_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX, "benchmark", "run_pytorch_training_performance_gpu_synthetic",
)
PT_PERFORMANCE_TRAINING_GPU_IMAGENET_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX, "benchmark", "run_pytorch_training_performance_gpu_imagenet"
)

PT_EC2_GPU_SYNTHETIC_INSTANCE_TYPE = "p3.16xlarge"
PT_EC2_GPU_IMAGENET_INSTANCE_TYPE = "p3.16xlarge"
PT_EC2_HPU_INSTANCE_TYPE = "dl1.24xlarge"

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

@pytest.mark.model("resnet50")
@pytest.mark.parametrize("ec2_instance_type", [PT_EC2_HPU_INSTANCE_TYPE], indirect=True)
@pytest.mark.parametrize("ec2_instance_ami", [HPU_AL2_DLAMI], indirect=True)
@pytest.mark.parametrize('cards_num', [1, 8])
def test_performance_pytorch_rn50_hpu_synthetic(pytorch_training_habana, ec2_connection, cards_num):
    _, framework_version = get_framework_and_version_from_tag(pytorch_training_habana)
    synapseai_version = get_synapseai_version_from_tag(pytorch_training_habana)
    threshold_1 = get_threshold_for_image(framework_version, PYTORCH_TRAINING_RN50_HPU_SYNTHETIC_1_CARD_THRESHOLD)
    threshold_8 = get_threshold_for_image(framework_version, PYTORCH_TRAINING_RN50_HPU_SYNTHETIC_8_CARD_THRESHOLD)
    expected_perf = (threshold_8/8) if cards_num > 1 else threshold_1    # Logs show per card performance for multicard
    allowed_regression = expected_perf * 0.1
    threshold = expected_perf - allowed_regression
    execute_ec2_training_performance_test(
        ec2_connection,
        pytorch_training_habana,
        PT_PERFORMANCE_RN50_TRAINING_HPU_SYNTHETIC_CMD,
        post_process=post_process_pytorch_hpu_py3_synthetic_ec2_training_performance,
        data_source="synthetic",
        threshold={"Throughput": threshold},
        cards_num=cards_num,
        synapseai_version=synapseai_version
    )

@pytest.mark.model("bert")
@pytest.mark.parametrize("ec2_instance_type", [PT_EC2_HPU_INSTANCE_TYPE], indirect=True)
@pytest.mark.parametrize("ec2_instance_ami", [HPU_AL2_DLAMI], indirect=True)
@pytest.mark.parametrize('cards_num', [1, 8])
def test_performance_pytorch_bert_hpu(pytorch_training_habana, ec2_connection, cards_num):
    _, framework_version = get_framework_and_version_from_tag(pytorch_training_habana)
    synapseai_version = get_synapseai_version_from_tag(pytorch_training_habana)
    threshold = get_threshold_for_image(framework_version, PYTORCH_TRAINING_BERT_HPU_THRESHOLD)
    perf_factor = 0.95 if cards_num > 1 else 1  # Rough scaling factor
    expected_perf = threshold * perf_factor
    allowed_regression = expected_perf * 0.1
    threshold = expected_perf - allowed_regression
    execute_ec2_training_performance_test(
        ec2_connection,
        pytorch_training_habana,
        PT_PERFORMANCE_BERT_TRAINING_HPU_CMD,
        post_process=post_process_pytorch_hpu_py3_synthetic_ec2_training_performance,
        data_source="synthetic",
        threshold={"Throughput": threshold},
        cards_num=cards_num,
        synapseai_version=synapseai_version
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

def post_process_pytorch_hpu_py3_synthetic_ec2_training_performance(connection, log_location):
    log_lines = connection.run(f"tail -n 20 {log_location}").stdout.split("\n")
    throughput = 0
    for line in reversed(log_lines):
        if "Validating result: actual" in line:
            throughput = float(line.split(" ")[6].strip(","))
            break
    return {"Throughput": throughput}

def post_process_pytorch_gpu_py3_synthetic_ec2_training_performance(connection, log_location):
    last_lines = connection.run(f"tail -n 20 {log_location}").stdout.split("\n")
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
