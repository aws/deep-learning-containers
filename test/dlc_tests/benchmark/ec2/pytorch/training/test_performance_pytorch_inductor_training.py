import os
import time
import pytest
from packaging.version import Version

from test.test_utils import (
    CONTAINER_TESTS_PREFIX,
    UBUNTU_18_BASE_DLAMI_US_WEST_2,
    DEFAULT_REGION,
    get_framework_and_version_from_tag,
    is_pr_context,
)

PT_PERFORMANCE_TRAINING_GPU_INDUCTOR_HUGGINGFACE_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX, "benchmark", "run_pytorch_inductor_training_benchmark_gpu_huggingface"
)
PT_PERFORMANCE_TRAINING_GPU_INDUCTOR_TIMM_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX, "benchmark", "run_pytorch_inductor_training_benchmark_gpu_timm"
)
PT_PERFORMANCE_TRAINING_GPU_INDUCTOR_TORCHBENCH_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX, "benchmark", "run_pytorch_inductor_training_benchmark_gpu_torchbench"
)

PT_EC2_GPU_INDUCTOR_INSTANCE_TYPES = ["p3.2xlarge", "p4d.24xlarge", "g5.4xlarge", "g4dn.4xlarge"]
PT_EC2_GPU_INDUCTOR_INSTANCE_TYPE_P3 = ["p3.2xlarge"]
PT_EC2_GPU_INDUCTOR_INSTANCE_TYPE_P4D = ["p4d.24xlarge"]
PT_EC2_GPU_INDUCTOR_INSTANCE_TYPE_G5 = ["g5.4xlarge"]
PT_EC2_GPU_INDUCTOR_INSTANCE_TYPE_G4DN = ["g4dn.4xlarge"]



@pytest.mark.integration("inductor")
@pytest.mark.model("huggingface")
@pytest.mark.parametrize("ec2_instance_ami", [UBUNTU_18_BASE_DLAMI_US_WEST_2], indirect=True)
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INDUCTOR_INSTANCE_TYPE_P3, indirect=True)
def test_performance_pytorch_gpu_inductor_huggingface_p3(pytorch_training, ec2_connection, gpu_only, py3_only, ec2_instance_type):
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_training)
    if Version(image_framework_version) < Version("2.0"):
        pytest.skip("Torch inductor was introduced in PyTorch 2.0")
    test_cmd = PT_PERFORMANCE_TRAINING_GPU_INDUCTOR_HUGGINGFACE_CMD + " " + ec2_instance_type
    execute_ec2_training_performance_test(
        ec2_connection, pytorch_training, test_cmd, model_suite="huggingface"
    )

@pytest.mark.integration("inductor")
@pytest.mark.model("huggingface")
@pytest.mark.parametrize("ec2_instance_ami", [UBUNTU_18_BASE_DLAMI_US_WEST_2], indirect=True)
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INDUCTOR_INSTANCE_TYPE_P4D, indirect=True)
def test_performance_pytorch_gpu_inductor_huggingfac_p4d(pytorch_training, ec2_connection, gpu_only, py3_only, ec2_instance_type):
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_training)
    if Version(image_framework_version) < Version("2.0"):
        pytest.skip("Torch inductor was introduced in PyTorch 2.0")
    test_cmd = PT_PERFORMANCE_TRAINING_GPU_INDUCTOR_HUGGINGFACE_CMD + " " + ec2_instance_type
    execute_ec2_training_performance_test(
        ec2_connection, pytorch_training, test_cmd, model_suite="huggingface"
    )

@pytest.mark.integration("inductor")
@pytest.mark.model("huggingface")
@pytest.mark.parametrize("ec2_instance_ami", [UBUNTU_18_BASE_DLAMI_US_WEST_2], indirect=True)
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INDUCTOR_INSTANCE_TYPE_G5, indirect=True)
def test_performance_pytorch_gpu_inductor_huggingface_g5(pytorch_training, ec2_connection, gpu_only, py3_only, ec2_instance_type):
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_training)
    if Version(image_framework_version) < Version("2.0"):
        pytest.skip("Torch inductor was introduced in PyTorch 2.0")
    test_cmd = PT_PERFORMANCE_TRAINING_GPU_INDUCTOR_HUGGINGFACE_CMD + " " + ec2_instance_type
    execute_ec2_training_performance_test(
        ec2_connection, pytorch_training, test_cmd, model_suite="huggingface"
    )

@pytest.mark.integration("inductor")
@pytest.mark.model("huggingface")
@pytest.mark.parametrize("ec2_instance_ami", [UBUNTU_18_BASE_DLAMI_US_WEST_2], indirect=True)
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INDUCTOR_INSTANCE_TYPE_G4DN, indirect=True)
def test_performance_pytorch_gpu_inductor_huggingface_g4dn(pytorch_training, ec2_connection, gpu_only, py3_only, ec2_instance_type):
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_training)
    if Version(image_framework_version) < Version("2.0"):
        pytest.skip("Torch inductor was introduced in PyTorch 2.0")
    test_cmd = PT_PERFORMANCE_TRAINING_GPU_INDUCTOR_HUGGINGFACE_CMD + " " + ec2_instance_type
    execute_ec2_training_performance_test(
        ec2_connection, pytorch_training, test_cmd, model_suite="huggingface"
    )
    
@pytest.mark.skip("skip for now")
@pytest.mark.integration("inductor")
@pytest.mark.model("timm")
@pytest.mark.parametrize("ec2_instance_ami", [UBUNTU_18_BASE_DLAMI_US_WEST_2], indirect=True)
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INDUCTOR_INSTANCE_TYPES, indirect=True)
def test_performance_pytorch_gpu_inductor_timm(pytorch_training, ec2_connection, gpu_only, py3_only, ec2_instance_type):
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_training)
    if Version(image_framework_version) < Version("2.0"):
        pytest.skip("Torch inductor was introduced in PyTorch 2.0")
    execute_ec2_training_performance_test(
        ec2_connection, pytorch_training, PT_PERFORMANCE_TRAINING_GPU_INDUCTOR_TIMM_CMD, model_suite="timm"
    )

@pytest.mark.skip("skip for now")
@pytest.mark.integration("inductor")
@pytest.mark.model("torchbench")
@pytest.mark.parametrize("ec2_instance_ami", [UBUNTU_18_BASE_DLAMI_US_WEST_2], indirect=True)
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INDUCTOR_INSTANCE_TYPES, indirect=True)
def test_performance_pytorch_gpu_inductor_torchbench(pytorch_training, ec2_connection, gpu_only, py3_only, ec2_instance_type):
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_training)
    if Version(image_framework_version) < Version("2.0"):
        pytest.skip("Torch inductor was introduced in PyTorch 2.0")
    execute_ec2_training_performance_test(
        ec2_connection, pytorch_training, PT_PERFORMANCE_TRAINING_GPU_INDUCTOR_TORCHBENCH_CMD, model_suite="torchbench"
    )


def execute_ec2_training_performance_test(
    connection, ecr_uri, test_cmd, model_suite, region=DEFAULT_REGION):
    docker_cmd = "nvidia-docker" if "gpu" in ecr_uri else "docker"
    container_test_local_dir = os.path.join("$HOME", "container_tests")

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    log_name = f"{model_suite}_results_{os.getenv('CODEBUILD_RESOLVED_SOURCE_VERSION')}_{timestamp}.txt"
    log_location = os.path.join(container_test_local_dir, "benchmark", "logs", log_name)

    # Make sure we are logged into ECR so we can pull the image
    connection.run(f"$(aws ecr get-login --no-include-email --region {region})", hide=True)
    connection.run(f"{docker_cmd} pull {ecr_uri}", hide='out') # this line fixes the time out issue

    # Run training command
    connection.run(
        f"{docker_cmd} run --user root "
        f"-e LOG_FILE={os.path.join(os.sep, 'test', 'benchmark', 'logs', log_name)} "
        f"-e PR_CONTEXT={1 if is_pr_context() else 0} "
        f"-v {container_test_local_dir}:{os.path.join(os.sep, 'test')} {ecr_uri} "
        f"{os.path.join(os.sep, 'bin', 'bash')} -c {test_cmd}"
    )
