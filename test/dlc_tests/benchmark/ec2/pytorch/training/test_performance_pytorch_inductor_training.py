import os
import re
import time
import pytest
import subprocess
import logging
from packaging.version import Version

from test.test_utils import (
    CONTAINER_TESTS_PREFIX,
    UBUNTU_18_BASE_DLAMI_US_WEST_2,
    DEFAULT_REGION,
    get_framework_and_version_from_tag,
    is_pr_context,
    BENCHMARK_RESULTS_S3_BUCKET_TRCOMP,
)

from test.test_utils.ec2 import (
    read_metric,
    put_metric_data,
)

PT_EC2_GPU_INDUCTOR_INSTANCE_TYPES = ["p3.2xlarge", "p4d.24xlarge", "g5.4xlarge", "g4dn.4xlarge"]
METRIC_NAMES = ["speedup", "comp_time", "memory", "passrate"]

@pytest.mark.integration("inductor")
@pytest.mark.model("huggingface")
@pytest.mark.parametrize("ec2_instance_ami", [UBUNTU_18_BASE_DLAMI_US_WEST_2], indirect=True)
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INDUCTOR_INSTANCE_TYPES, indirect=True)
def test_performance_pytorch_gpu_inductor_huggingface(pytorch_training, ec2_connection, gpu_only, py3_only, ec2_instance_type):
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_training)
    if Version(image_framework_version) < Version("2.0"):
        pytest.skip("Torch inductor was introduced in PyTorch 2.0")
    execute_ec2_training_performance_test(
        ec2_connection, pytorch_training, "huggingface", ec2_instance_type)


@pytest.mark.integration("inductor")
@pytest.mark.model("timm_models")
@pytest.mark.parametrize("ec2_instance_ami", [UBUNTU_18_BASE_DLAMI_US_WEST_2], indirect=True)
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INDUCTOR_INSTANCE_TYPES, indirect=True)
def test_performance_pytorch_gpu_inductor_timm_models(pytorch_training, ec2_connection, gpu_only, py3_only, ec2_instance_type):
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_training)
    if Version(image_framework_version) < Version("2.0"):
        pytest.skip("Torch inductor was introduced in PyTorch 2.0")
    if ec2_instance_type == "g4dn.4xlarge":
        pytest.skip("This takes more than 4 hours to complete so skip for now")
    execute_ec2_training_performance_test(
        ec2_connection, pytorch_training, "timm_models", ec2_instance_type)


@pytest.mark.integration("inductor")
@pytest.mark.model("torchbench")
@pytest.mark.parametrize("ec2_instance_ami", [UBUNTU_18_BASE_DLAMI_US_WEST_2], indirect=True)
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INDUCTOR_INSTANCE_TYPES, indirect=True)
def test_performance_pytorch_gpu_inductor_torchbench(pytorch_training, ec2_connection, gpu_only, py3_only, ec2_instance_type):
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_training)
    if Version(image_framework_version) < Version("2.0"):
        pytest.skip("Torch inductor was introduced in PyTorch 2.0")
    execute_ec2_training_performance_test(
        ec2_connection, pytorch_training, "torchbench", ec2_instance_type)

def execute_ec2_training_performance_test(
    connection, ecr_uri, model_suite, ec2_instance_type, region=DEFAULT_REGION):
    _, image_framework_version = get_framework_and_version_from_tag(ecr_uri)
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    s3_key = os.path.join(os.sep, model_suite, re.sub("\.", "-", ec2_instance_type), timestamp)
    s3_pth = BENCHMARK_RESULTS_S3_BUCKET_TRCOMP + s3_key
    test_prefix = os.path.join(CONTAINER_TESTS_PREFIX, "benchmark", f"run_pytorch_inductor_training_benchmark_gpu_{model_suite}")   
    test_cmd = test_prefix + " " + ec2_instance_type + " " + s3_pth

    docker_cmd = "nvidia-docker" if "gpu" in ecr_uri else "docker"
    container_test_local_dir = os.path.join("$HOME", "container_tests")

    log_name = f"{model_suite}_results_{os.getenv('CODEBUILD_RESOLVED_SOURCE_VERSION')}_{timestamp}.txt"
    log_location = os.path.join(os.sep, "test", "benchmark", "logs", log_name)

    # Make sure we are logged into ECR so we can pull the image
    connection.run(f"$(aws ecr get-login --no-include-email --region {region})", hide=True)
    connection.run(f"{docker_cmd} pull {ecr_uri}", hide='out') # this line fixes the time out issue
    # Run training command
    try:
        connection.run(
            f"{docker_cmd} run --user root "
            f"-e LOG_FILE={os.path.join(os.sep, 'test', 'benchmark', 'logs', log_name)} "
            f"-e PR_CONTEXT={1 if is_pr_context() else 0} "
            f"-v {container_test_local_dir}:{os.path.join(os.sep, 'test')} {ecr_uri} "
            f"{os.path.join(os.sep, 'bin', 'bash')} {test_cmd}", timeout=300)
    finally:
        subprocess.check_output(f"rm -rf {model_suite}", shell=True)
        subprocess.check_output(f"mkdir {model_suite}", shell=True)
        subprocess.check_output(f"aws s3 cp {s3_pth}/ {model_suite}/ --recursive", shell=True)
        try:
            read_upload_benchmarking_result_to_cw(METRIC_NAMES, model_suite, instance_type=ec2_instance_type, model_suite=model_suite)
        except:
            print("Benchmarking was not finished in 5 minutes.")


def read_upload_benchmarking_result_to_cw(metric_names, pth, instance_type="p4d.24xlarge", model_suite="huggingface", precision="amp", namespace="PyTorch/EC2/Benchmarks/TorchDynamo/Inductor"):
    
    dimensions = [
             {"Name": "InstanceType", "Value": instance_type},
             {"Name": "ModelSuite", "Value": model_suite},
             {"Name": "Precision", "Value": precision},
             {"Name": "WorkLoad", "Value": "Training"},
         ]
    for name in metric_names:
        if name == "speedup":
            value = read_metric(os.path.join(pth, "geomean.csv"))
            unit = "None"
        if name == "comp_time":
            value = read_metric(os.path.join(pth, "comp_time.csv"))
            unit = "Seconds"
        if name == "memory":
            value = read_metric(os.path.join(pth, "memory.csv"))
            unit = "None"
        if name == "pass_rate":
            value = read_metric(os.path.join(pth, "pass_rate.csv"))
            unit = "Percent"

        put_metric_data(name, namespace, unit, value, dimensions)

