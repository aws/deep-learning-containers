import json
import logging
import os
import re
import subprocess
import sys
import time

import boto3
import git
import pytest

import boto3
from botocore.exceptions import ClientError
from glob import glob
from invoke import run
from invoke.context import Context
from packaging.version import LegacyVersion, Version, parse
from packaging.specifiers import SpecifierSet
from retrying import retry

from src import config

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stderr))

# Constant to represent default region for boto3 commands
DEFAULT_REGION = "us-west-2"
# Constant to represent region where p3dn tests can be run
P3DN_REGION = "us-east-1"

UBUNTU_18_BASE_DLAMI_US_WEST_2 = "ami-0150e36b3f936a26e"
UBUNTU_18_BASE_DLAMI_US_EAST_1 = "ami-044971d381e6a1109"
AML2_GPU_DLAMI_US_WEST_2 = "ami-071cb1e434903a577"
AML2_GPU_DLAMI_US_EAST_1 = "ami-044264d246686b043"
AML2_CPU_ARM64_US_WEST_2 = "ami-0bccd90b9db95e2e5"
UL18_CPU_ARM64_US_WEST_2 = "ami-00bccef9d47441ac9"
UL18_BENCHMARK_CPU_ARM64_US_WEST_2 = "ami-0ababa2deb802b069"
AML2_CPU_ARM64_US_EAST_1 = "ami-01c47f32b27ed7fa0"
PT_GPU_PY3_BENCHMARK_IMAGENET_AMI_US_EAST_1 = "ami-0673bb31cc62485dd"
PT_GPU_PY3_BENCHMARK_IMAGENET_AMI_US_WEST_2 = "ami-02d9a47bc61a31d43"
# Since latest driver is not in public DLAMI yet, using a custom one
NEURON_UBUNTU_18_BASE_DLAMI_US_WEST_2 = "ami-078c2404eecfbe916"
# Habana Base v1.3 ami
UBUNTU_18_HPU_DLAMI_US_WEST_2 = "ami-0ef18b1906e7010fb"
UBUNTU_18_HPU_DLAMI_US_EAST_1 = "ami-040ef14d634e727a2"
UL_AMI_LIST = [
    UBUNTU_18_BASE_DLAMI_US_EAST_1,
    UBUNTU_18_BASE_DLAMI_US_WEST_2,
    UBUNTU_18_HPU_DLAMI_US_WEST_2,
    UBUNTU_18_HPU_DLAMI_US_EAST_1,
    PT_GPU_PY3_BENCHMARK_IMAGENET_AMI_US_EAST_1,
    PT_GPU_PY3_BENCHMARK_IMAGENET_AMI_US_WEST_2,
    NEURON_UBUNTU_18_BASE_DLAMI_US_WEST_2,
    UL18_CPU_ARM64_US_WEST_2,
    UL18_BENCHMARK_CPU_ARM64_US_WEST_2,
]

# ECS images are maintained here: https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-optimized_AMI.html
ECS_AML2_GPU_USWEST2 = "ami-052237b2eee880dc6"
ECS_AML2_CPU_USWEST2 = "ami-0794df61693647a62"
ECS_AML2_NEURON_USWEST2 = "ami-0c7321fe2b2340dd5"
ECS_AML2_GRAVITON_CPU_USWEST2 = "ami-0fb32cf53e5ab7686"
NEURON_AL2_DLAMI = "ami-03c4cdc89eca4dbcb"
HPU_AL2_DLAMI = "ami-052f4f716a7c7bad7"

# S3 bucket for TensorFlow models
TENSORFLOW_MODELS_BUCKET = "s3://tensoflow-trained-models"

DLAMI_PYTHON_MAPPING = {
    UBUNTU_18_BASE_DLAMI_US_WEST_2: "/usr/bin/python3.7",
    UBUNTU_18_BASE_DLAMI_US_EAST_1: "/usr/bin/python3.7",
    UL18_CPU_ARM64_US_WEST_2: "/usr/bin/python3.8",
}
# Used for referencing tests scripts from container_tests directory (i.e. from ECS cluster)
CONTAINER_TESTS_PREFIX = os.path.join(os.sep, "test", "bin")

# S3 Bucket to use to transfer tests into an EC2 instance
TEST_TRANSFER_S3_BUCKET = "s3://dlinfra-tests-transfer-bucket"

# S3 Bucket to use to record benchmark results for further retrieving
BENCHMARK_RESULTS_S3_BUCKET = "s3://dlinfra-dlc-cicd-performance"

# Ubuntu ami home dir
UBUNTU_HOME_DIR = "/home/ubuntu"

# Reason string for skipping tests in PR context
SKIP_PR_REASON = "Skipping test in PR context to speed up iteration time. Test will be run in nightly/release pipeline."

# Reason string for skipping tests in non-PR context
PR_ONLY_REASON = "Skipping test that doesn't need to be run outside of PR context."

KEYS_TO_DESTROY_FILE = os.path.join(os.sep, "tmp", "keys_to_destroy.txt")

# Sagemaker test types
SAGEMAKER_LOCAL_TEST_TYPE = "local"
SAGEMAKER_REMOTE_TEST_TYPE = "sagemaker"

PUBLIC_DLC_REGISTRY = "763104351884"

SAGEMAKER_EXECUTION_REGIONS = ["us-west-2", "us-east-1", "eu-west-1"]

UPGRADE_ECR_REPO_NAME = "upgraded-image-ecr-scan-repo"
ECR_SCAN_HELPER_BUCKET = f"""ecr-scan-helper-{boto3.client("sts", region_name=DEFAULT_REGION).get_caller_identity().get("Account")}"""
ECR_SCAN_FAILURE_ROUTINE_LAMBDA = "ecr-scan-failure-routine-lambda"

class MissingPythonVersionException(Exception):
    """
    When the Python Version is missing from an image_uri where it is expected to exist
    """

    pass

class CudaVersionTagNotFoundException(Exception):
    """
    When none of the tags of a GPU image have a Cuda version in them
    """

    pass


def get_dockerfile_path_for_image(image_uri):
    """
    For a given image_uri, find the path within the repository to its corresponding dockerfile

    :param image_uri: str Image URI
    :return: str Absolute path to dockerfile
    """
    github_repo_path = os.path.abspath(os.path.curdir).split("test", 1)[0]

    framework, framework_version = get_framework_and_version_from_tag(image_uri)

    if "huggingface" in framework:
        framework_path = framework.replace("_", os.path.sep)
    elif "habana" in image_uri:
        framework_path = os.path.join("habana", framework)
    else:
        framework_path = framework

    job_type = get_job_type_from_image(image_uri)

    short_framework_version = re.search(r"(\d+\.\d+)", image_uri).group(1)

    framework_version_path = os.path.join(github_repo_path, framework_path, job_type, "docker", short_framework_version)
    if not os.path.isdir(framework_version_path):
        long_framework_version = re.search(r"\d+(\.\d+){2}", image_uri).group()
        framework_version_path = os.path.join(
            github_repo_path, framework_path, job_type, "docker", long_framework_version
        )
    python_version = re.search(r"py\d+", image_uri).group()

    python_version_path = os.path.join(framework_version_path, python_version)
    if not os.path.isdir(python_version_path):
        python_version_path = os.path.join(framework_version_path, "py3")

    device_type = get_processor_from_image_uri(image_uri)
    cuda_version = get_cuda_version_from_tag(image_uri)
    synapseai_version = get_synapseai_version_from_tag(image_uri)
    neuron_sdk_version = get_neuron_sdk_version_from_tag(image_uri)

    dockerfile_name = get_expected_dockerfile_filename(device_type, image_uri)

    dockerfiles_list = [
        path
        for path in glob(os.path.join(python_version_path, "**", dockerfile_name), recursive=True)
        if "example" not in path
    ]

    if device_type in ["gpu", "hpu", "neuron"]:
        if len(dockerfiles_list) > 1:
            if device_type == "gpu" and not cuda_version:
                raise LookupError(
                    f"dockerfiles_list has more than one result, and needs cuda_version to be in image_uri to "
                    f"uniquely identify the right dockerfile:\n"
                    f"{dockerfiles_list}"
                )
            if device_type == "hpu" and not synapseai_version:
                raise LookupError(
                    f"dockerfiles_list has more than one result, and needs synapseai_version to be in image_uri to "
                    f"uniquely identify the right dockerfile:\n"
                    f"{dockerfiles_list}"
                )
            if device_type == "neuron" and not neuron_sdk_version:
                raise LookupError(
                    f"dockerfiles_list has more than one result, and needs neuron_sdk_version to be in image_uri to "
                    f"uniquely identify the right dockerfile:\n"
                    f"{dockerfiles_list}"
                )
        for dockerfile_path in dockerfiles_list:
            if cuda_version:
                if cuda_version in dockerfile_path:
                    return dockerfile_path
            elif synapseai_version:
                if synapseai_version in dockerfile_path:
                    return dockerfile_path
            elif neuron_sdk_version:
                if neuron_sdk_version in dockerfile_path:
                    return dockerfile_path
        raise LookupError(f"Failed to find a dockerfile path for {cuda_version} in:\n{dockerfiles_list}")

    assert len(dockerfiles_list) == 1, f"No unique dockerfile path in:\n{dockerfiles_list}\nfor image: {image_uri}"

    return dockerfiles_list[0]


def get_expected_dockerfile_filename(device_type, image_uri):
    if is_e3_image(image_uri):
        return f"Dockerfile.e3.{device_type}"
    if is_sagemaker_image(image_uri):
        return f"Dockerfile.sagemaker.{device_type}"
    return f"Dockerfile.{device_type}"


def get_customer_type():
    return os.getenv("CUSTOMER_TYPE")


def get_python_invoker(ami_id):
    return DLAMI_PYTHON_MAPPING.get(ami_id, "/usr/bin/python3")


def get_ecr_repo_name(image_uri):
    """
    Retrieve ECR repository name from image URI
    :param image_uri: str ECR Image URI
    :return: str ECR repository name
    """
    ecr_repo_name = image_uri.split("/")[-1].split(":")[0]
    return ecr_repo_name


def is_tf_version(required_version, image_uri):
    """
    Validate that image_uri has framework version equal to required_version

    :param required_version: str Framework version which is required from the image_uri
    :param image_uri: str ECR Image URI for the image to be validated
    :return: bool True if image_uri has same framework version as required_version, else False
    """
    image_framework_name, image_framework_version = get_framework_and_version_from_tag(image_uri)
    required_version_specifier_set = SpecifierSet(f"=={required_version}.*")
    return image_framework_name == "tensorflow" and image_framework_version in required_version_specifier_set


def is_below_framework_version(version_upper_bound, image_uri, framework):
    """
    Validate that image_uri has framework version strictly less than version_upper_bound

    :param version_upper_bound: str Framework version that image_uri is required to be below
    :param image_uri: str ECR Image URI for the image to be validated
    :return: bool True if image_uri has framework version less than version_upper_bound, else False
    """
    image_framework_name, image_framework_version = get_framework_and_version_from_tag(image_uri)
    required_version_specifier_set = SpecifierSet(f"<{version_upper_bound}")
    return image_framework_name == framework and image_framework_version in required_version_specifier_set


def is_image_incompatible_with_instance_type(image_uri, ec2_instance_type):
    """
    Check for all compatibility issues between DLC Image Types and EC2 Instance Types.
    Currently configured to fail on the following checks:
        1. p4d.24xlarge instance type is used with a cuda<11.0 image
        2. p2.8xlarge instance type is used with a cuda=11.0 image for MXNET framework

    :param image_uri: ECR Image URI in valid DLC-format
    :param ec2_instance_type: EC2 Instance Type
    :return: bool True if there are incompatibilities, False if there aren't
    """
    incompatible_conditions = []
    framework, framework_version = get_framework_and_version_from_tag(image_uri)

    image_is_cuda10_on_incompatible_p4d_instance = (
        get_processor_from_image_uri(image_uri) == "gpu"
        and get_cuda_version_from_tag(image_uri).startswith("cu10")
        and ec2_instance_type in ["p4d.24xlarge"]
    )
    incompatible_conditions.append(image_is_cuda10_on_incompatible_p4d_instance)

    image_is_cuda11_on_incompatible_p2_instance_mxnet = (
        framework == "mxnet"
        and get_processor_from_image_uri(image_uri) == "gpu"
        and get_cuda_version_from_tag(image_uri).startswith("cu11")
        and ec2_instance_type in ["p2.8xlarge"]
    )
    incompatible_conditions.append(image_is_cuda11_on_incompatible_p2_instance_mxnet)

    image_is_pytorch_1_11_on_incompatible_p2_instance_pytorch = (
        framework == "pytorch"
        and Version(framework_version) in SpecifierSet("==1.11.*")
        and get_processor_from_image_uri(image_uri) == "gpu"
        and ec2_instance_type in ["p2.8xlarge"]
    )
    incompatible_conditions.append(image_is_pytorch_1_11_on_incompatible_p2_instance_pytorch)

    return any(incompatible_conditions)


def get_repository_local_path():
    git_repo_path = os.getcwd().split("/test/")[0]
    return git_repo_path


def get_inference_server_type(image_uri):
    if "pytorch" not in image_uri:
        return "mms"
    if "neuron" in image_uri:
        return "ts"
    image_tag = image_uri.split(":")[1]
    pytorch_ver = parse(image_tag.split("-")[0])
    if isinstance(pytorch_ver, LegacyVersion) or pytorch_ver < Version("1.6"):
        return "mms"
    return "ts"


def get_build_context():
    return os.getenv("BUILD_CONTEXT")


def is_pr_context():
    return os.getenv("BUILD_CONTEXT") == "PR"


def is_canary_context():
    return os.getenv("BUILD_CONTEXT") == "CANARY"


def is_mainline_context():
    return os.getenv("BUILD_CONTEXT") == "MAINLINE"


def is_nightly_context():
    return os.getenv("BUILD_CONTEXT") == "NIGHTLY"


def is_empty_build_context():
    return not os.getenv("BUILD_CONTEXT")


def is_graviton_architecture():
    return os.getenv("ARCH_TYPE") == "graviton"


def is_dlc_cicd_context():
    return os.getenv("BUILD_CONTEXT") in ["PR", "CANARY", "NIGHTLY", "MAINLINE"]


def is_benchmark_dev_context():
    return config.is_benchmark_mode_enabled()


def is_rc_test_context():
    sm_remote_tests_val = config.get_sagemaker_remote_tests_config_value()
    return sm_remote_tests_val == config.AllowedSMRemoteConfigValues.RC.value


def is_e3_image(image_uri):
    return "-e3" in image_uri


def is_sagemaker_image(image_uri):
    return "-sagemaker" in image_uri


def is_time_for_canary_safety_scan():
    """
    Canary tests run every 15 minutes.
    Using a 20 minutes interval to make tests run only once a day around 9 am PST (10 am during winter time).
    """
    current_utc_time = time.gmtime()
    return current_utc_time.tm_hour == 16 and (0 < current_utc_time.tm_min < 20)

def is_time_for_invoking_ecr_scan_failure_routine_lambda():
    """
    Canary tests run every 15 minutes.
    Using a 20 minutes interval to make tests run only once a day around 9 am PST (10 am during winter time).
    """
    current_utc_time = time.gmtime()
    return current_utc_time.tm_hour == 16 and (0 < current_utc_time.tm_min < 20)


def _get_remote_override_flags():
    try:
        s3_client = boto3.client("s3")
        sts_client = boto3.client("sts")
        account_id = sts_client.get_caller_identity().get("Account")
        result = s3_client.get_object(Bucket=f"dlc-cicd-helper-{account_id}", Key="override_tests_flags.json")
        json_content = json.loads(result["Body"].read().decode("utf-8"))
    except ClientError as e:
        LOGGER.warning("ClientError when performing S3/STS operation: {}".format(e))
        json_content = {}
    return json_content


# Now we can skip EFA tests on pipeline without making any source code change
def are_efa_tests_disabled():
    disable_efa_tests = is_pr_context() and os.getenv("DISABLE_EFA_TESTS", "False").lower() == "true"

    remote_override_flags = _get_remote_override_flags()
    override_disable_efa_tests = remote_override_flags.get("disable_efa_tests", "false").lower() == "true"

    return disable_efa_tests or override_disable_efa_tests


def is_safety_test_context():
    return config.is_safety_check_test_enabled()


def is_test_disabled(test_name, build_name, version):
    """
    Expected format of remote_override_flags:
    {
        "CB Project Name for Test Type A": {
            "CodeBuild Resolved Source Version": ["test_type_A_test_function_1", "test_type_A_test_function_2"]
        },
        "CB Project Name for Test Type B": {
            "CodeBuild Resolved Source Version": ["test_type_B_test_function_1", "test_type_B_test_function_2"]
        }
    }

    :param test_name: str Test Function node name (includes parametrized values in string)
    :param build_name: str Build Project name of current execution
    :param version: str Source Version of current execution
    :return: bool True if test is disabled as per remote override, False otherwise
    """
    remote_override_flags = _get_remote_override_flags()
    remote_override_build = remote_override_flags.get(build_name, {})
    if version in remote_override_build:
        return not remote_override_build[version] or any(
            [test_keyword in test_name for test_keyword in remote_override_build[version]]
        )
    return False


def run_subprocess_cmd(cmd, failure="Command failed"):
    command = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
    if command.returncode:
        pytest.fail(f"{failure}. Error log:\n{command.stdout.decode()}")
    return command


def login_to_ecr_registry(context, account_id, region):
    """
    Function to log into an ecr registry

    :param context: either invoke context object or fabric connection object
    :param account_id: Account ID with the desired ecr registry
    :param region: i.e. us-west-2
    """
    context.run(
        f"aws ecr get-login-password --region {region} | docker login --username AWS "
        f"--password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com"
    )


def retry_if_result_is_false(result):
    """Return True if we should retry (in this case retry if the result is False), False otherwise"""
    return result is False


@retry(
    stop_max_attempt_number=10,
    wait_fixed=10000,
    retry_on_result=retry_if_result_is_false,
)
def request_mxnet_inference(ip_address="127.0.0.1", port="80", connection=None, model="squeezenet"):
    """
    Send request to container to test inference on kitten.jpg
    :param ip_address:
    :param port:
    :connection: ec2_connection object to run the commands remotely over ssh
    :return: <bool> True/False based on result of inference
    """
    conn_run = connection.run if connection is not None else run

    # Check if image already exists
    run_out = conn_run("[ -f kitten.jpg ]", warn=True)
    if run_out.return_code != 0:
        conn_run("curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg", hide=True)

    run_out = conn_run(f"curl -X POST http://{ip_address}:{port}/predictions/{model} -T kitten.jpg", warn=True)

    # The run_out.return_code is not reliable, since sometimes predict request may succeed but the returned result
    # is 404. Hence the extra check.
    if run_out.return_code != 0 or "probability" not in run_out.stdout:
        return False

    return True


@retry(stop_max_attempt_number=10, wait_fixed=10000, retry_on_result=retry_if_result_is_false)
def request_mxnet_inference_gluonnlp(ip_address="127.0.0.1", port="80", connection=None):
    """
    Send request to container to test inference for predicting sentiments.
    :param ip_address:
    :param port:
    :connection: ec2_connection object to run the commands remotely over ssh
    :return: <bool> True/False based on result of inference
    """
    conn_run = connection.run if connection is not None else run
    run_out = conn_run(
        (
            f"curl -X POST http://{ip_address}:{port}/predictions/bert_sst/predict -F "
            '\'data=["Positive sentiment", "Negative sentiment"]\''
        ),
        warn=True,
    )

    # The run_out.return_code is not reliable, since sometimes predict request may succeed but the returned result
    # is 404. Hence the extra check.
    if run_out.return_code != 0 or "1" not in run_out.stdout:
        return False

    return True


@retry(
    stop_max_attempt_number=10,
    wait_fixed=10000,
    retry_on_result=retry_if_result_is_false,
)
def request_pytorch_inference_densenet(
    ip_address="127.0.0.1", port="80", connection=None, model_name="pytorch-densenet", server_type="ts"
):
    """
    Send request to container to test inference on flower.jpg
    :param ip_address: str
    :param port: str
    :param connection: obj
    :param model_name: str
    :return: <bool> True/False based on result of inference
    """
    conn_run = connection.run if connection is not None else run
    # Check if image already exists
    run_out = conn_run("[ -f flower.jpg ]", warn=True)
    if run_out.return_code != 0:
        conn_run("curl -O https://s3.amazonaws.com/model-server/inputs/flower.jpg", hide=True)

    run_out = conn_run(
        f"curl -X POST http://{ip_address}:{port}/predictions/{model_name} -T flower.jpg", hide=True, warn=True
    )

    # The run_out.return_code is not reliable, since sometimes predict request may succeed but the returned result
    # is 404. Hence the extra check.
    if run_out.return_code != 0:
        LOGGER.error("run_out.return_code != 0")
        return False
    else:
        inference_output = json.loads(run_out.stdout.strip("\n"))
        if not (
            ("neuron" in model_name and isinstance(inference_output, list) and len(inference_output) == 3)
            or (server_type == "ts" and isinstance(inference_output, dict) and len(inference_output) == 5)
            or (server_type == "mms" and isinstance(inference_output, list) and len(inference_output) == 5)
        ):
            return False
        LOGGER.info(f"Inference Output = {json.dumps(inference_output, indent=4)}")

    return True


@retry(stop_max_attempt_number=20, wait_fixed=10000, retry_on_result=retry_if_result_is_false)
def request_tensorflow_inference(
    model_name,
    ip_address="127.0.0.1",
    port="8501",
    inference_string="'{\"instances\": [1.0, 2.0, 5.0]}'",
    connection=None,
):
    """
    Method to run tensorflow inference on half_plus_two model using CURL command
    :param model_name:
    :param ip_address:
    :param port:
    :connection: ec2_connection object to run the commands remotely over ssh
    :return:
    """
    conn_run = connection.run if connection is not None else run
    run_out = conn_run(
        f"curl -d {inference_string} -X POST  http://{ip_address}:{port}/v1/models/{model_name}:predict", warn=True
    )

    # The run_out.return_code is not reliable, since sometimes predict request may succeed but the returned result
    # is 404. Hence the extra check.
    if run_out.return_code != 0 or "predictions" not in run_out.stdout:
        return False

    return True


@retry(stop_max_attempt_number=20, wait_fixed=10000, retry_on_result=retry_if_result_is_false)
def request_tensorflow_inference_nlp(model_name, ip_address="127.0.0.1", port="8501"):
    """
    Method to run tensorflow inference on half_plus_two model using CURL command
    :param model_name:
    :param ip_address:
    :param port:
    :connection: ec2_connection object to run the commands remotely over ssh
    :return:
    """
    inference_string = "'{\"instances\": [[2,1952,25,10901,3]]}'"
    run_out = run(
        f"curl -d {inference_string} -X POST http://{ip_address}:{port}/v1/models/{model_name}:predict", warn=True
    )

    # The run_out.return_code is not reliable, since sometimes predict request may succeed but the returned result
    # is 404. Hence the extra check.
    if run_out.return_code != 0 or "predictions" not in run_out.stdout:
        return False

    return True


def request_tensorflow_inference_grpc(
    script_file_path, ip_address="127.0.0.1", port="8500", connection=None, ec2_instance_ami=None
):
    """
    Method to run tensorflow inference on MNIST model using gRPC protocol
    :param script_file_path:
    :param ip_address:
    :param port:
    :param connection:
    :return:
    """
    python_invoker = get_python_invoker(ec2_instance_ami)
    conn_run = connection.run if connection is not None else run
    conn_run(f"{python_invoker} {script_file_path} --num_tests=1000 --server={ip_address}:{port}", hide=True)


def get_inference_run_command(image_uri, model_names, processor="cpu"):
    """
    Helper function to format run command for MMS
    :param image_uri:
    :param model_names:
    :param processor:
    :return: <str> Command to start MMS server with given model
    """
    server_type = get_inference_server_type(image_uri)
    if processor == "eia":
        multi_model_location = {
            "resnet-152-eia": "https://s3.amazonaws.com/model-server/model_archive_1.0/resnet-152-eia-1-7-0.mar",
            "resnet-152-eia-1-5-1": "https://s3.amazonaws.com/model-server/model_archive_1.0/resnet-152-eia-1-5-1.mar",
            "pytorch-densenet": "https://aws-dlc-sample-models.s3.amazonaws.com/pytorch/densenet_eia/densenet_eia_v1_5_1.mar",
            "pytorch-densenet-v1-3-1": "https://aws-dlc-sample-models.s3.amazonaws.com/pytorch/densenet_eia/densenet_eia_v1_3_1.mar",
        }
    elif server_type == "ts":
        multi_model_location = {
            "squeezenet": "https://torchserve.s3.amazonaws.com/mar_files/squeezenet1_1.mar",
            "pytorch-densenet": "https://torchserve.s3.amazonaws.com/mar_files/densenet161.mar",
            "pytorch-resnet-neuron": "https://aws-dlc-sample-models.s3.amazonaws.com/pytorch/Resnet50-neuron.mar",
        }
    else:
        multi_model_location = {
            "squeezenet": "https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model",
            "pytorch-densenet": "https://dlc-samples.s3.amazonaws.com/pytorch/multi-model-server/densenet/densenet.mar",
            "bert_sst": "https://aws-dlc-sample-models.s3.amazonaws.com/bert_sst/bert_sst.mar",
            "mxnet-resnet-neuron": "https://aws-dlc-sample-models.s3.amazonaws.com/mxnet/Resnet50-neuron.mar",
        }

    if not isinstance(model_names, list):
        model_names = [model_names]

    for model_name in model_names:
        if model_name not in multi_model_location:
            raise Exception("No entry found for model {} in dictionary".format(model_name))

    parameters = ["{}={}".format(name, multi_model_location[name]) for name in model_names]

    if server_type == "ts":
        server_cmd = "torchserve"
    else:
        server_cmd = "multi-model-server"

    if processor != "neuron":
        mms_command = (
            f"{server_cmd} --start --{server_type}-config /home/model-server/config.properties --models "
            + " ".join(parameters)
        )
    else:
        # Temp till the mxnet dockerfile also have the neuron entrypoint file
        if server_type == "ts":
            mms_command = (
                f"{server_cmd} --start --{server_type}-config /home/model-server/config.properties --models "
                + " ".join(parameters)
            )
        else:
            mms_command = f"/usr/local/bin/entrypoint.sh -t /home/model-server/config.properties -m " + " ".join(
                parameters
            )

    return mms_command


def get_tensorflow_model_name(processor, model_name):
    """
    Helper function to get tensorflow model name
    :param processor: Processor Type
    :param model_name: Name of model to be used
    :return: File name for model being used
    """
    tensorflow_models = {
        "saved_model_half_plus_two": {
            "cpu": "saved_model_half_plus_two_cpu",
            "gpu": "saved_model_half_plus_two_gpu",
            "eia": "saved_model_half_plus_two",
        },
        "albert": {
            "cpu": "albert",
            "gpu": "albert",
            "eia": "albert",
        },
        "saved_model_half_plus_three": {"eia": "saved_model_half_plus_three"},
        "simple": {"neuron": "simple"},
    }
    if model_name in tensorflow_models:
        return tensorflow_models[model_name][processor]
    else:
        raise Exception(f"No entry found for model {model_name} in dictionary")


def generate_ssh_keypair(ec2_client, key_name):
    pwd = run("pwd", hide=True).stdout.strip("\n")
    key_filename = os.path.join(pwd, f"{key_name}.pem")
    if os.path.exists(key_filename):
        run(f"chmod 400 {key_filename}")
        return key_filename
    try:
        key_pair = ec2_client.create_key_pair(KeyName=key_name)
    except ClientError as e:
        if "InvalidKeyPair.Duplicate" in f"{e}":
            # Wait 10 seconds for key to be created to avoid race condition
            time.sleep(10)
            if os.path.exists(key_filename):
                run(f"chmod 400 {key_filename}")
                return key_filename
        raise e

    run(f"echo '{key_pair['KeyMaterial']}' > {key_filename}")
    run(f"chmod 400 {key_filename}")
    return key_filename


def destroy_ssh_keypair(ec2_client, key_filename):
    key_name = os.path.basename(key_filename).split(".pem")[0]
    response = ec2_client.delete_key_pair(KeyName=key_name)
    run(f"rm -f {key_filename}")
    return response, key_name


def upload_tests_to_s3(testname_datetime_suffix):
    """
    Upload test-related artifacts to unique s3 location.
    Allows each test to have a unique remote location for test scripts and files.
    These uploaded files and folders are copied into a container running an ECS test.
    :param testname_datetime_suffix: test name and datetime suffix that is unique to a test
    :return: <bool> True if upload was successful, False if any failure during upload
    """
    s3_test_location = os.path.join(TEST_TRANSFER_S3_BUCKET, testname_datetime_suffix)
    run_out = run(f"aws s3 ls {s3_test_location}", warn=True)
    if run_out.return_code == 0:
        raise FileExistsError(f"{s3_test_location} already exists. Skipping upload and failing the test.")

    path = run("pwd", hide=True).stdout.strip("\n")
    if "dlc_tests" not in path:
        EnvironmentError("Test is being run from wrong path")
    while os.path.basename(path) != "dlc_tests":
        path = os.path.dirname(path)
    container_tests_path = os.path.join(path, "container_tests")

    run(f"aws s3 cp --recursive {container_tests_path}/ {s3_test_location}/")
    return s3_test_location


def delete_uploaded_tests_from_s3(s3_test_location):
    """
    Delete s3 bucket data related to current test after test is completed
    :param s3_test_location: S3 URI for test artifacts to be removed
    :return: <bool> True/False based on success/failure of removal
    """
    run(f"aws s3 rm --recursive {s3_test_location}")


def get_dlc_images():
    if is_pr_context() or is_empty_build_context():
        return os.getenv("DLC_IMAGES")
    elif is_canary_context():
        return parse_canary_images(os.getenv("FRAMEWORK"), os.getenv("AWS_REGION"))
    test_env_file = os.path.join(os.getenv("CODEBUILD_SRC_DIR_DLC_IMAGES_JSON"), "test_type_images.json")
    with open(test_env_file) as test_env:
        test_images = json.load(test_env)
    for dlc_test_type, images in test_images.items():
        if dlc_test_type == "sanity":
            return " ".join(images)
    raise RuntimeError(f"Cannot find any images for in {test_images}")


def get_canary_default_tag_py3_version(framework, version):
    """
    Currently, only TF2.2 images and above have major/minor python version in their canary tag. Creating this function
    to conditionally choose a python version based on framework version ranges. If we move up to py38, for example,
    this is the place to make the conditional change.

    :param framework: tensorflow1, tensorflow2, mxnet, pytorch
    :param version: fw major.minor version, i.e. 2.2
    :return: default tag python version
    """
    if framework == "tensorflow" or framework == "huggingface_tensorflow":
        if Version("2.2") <= Version(version) < Version("2.6"):
            return "py37"
        if Version("2.6") <= Version(version) < Version("2.8"):
            return "py38"
        if Version(version) >= Version("2.8"):
            return"py39"

    if framework == "mxnet":
        if Version(version) == Version("1.8"):
            return "py37"
        if Version(version) >= Version("1.9"):
            return "py38"

    if framework == "pytorch" or framework == "huggingface_pytorch":
        if Version(version) >= Version("1.9"):
            return "py38"

    return "py3"


def parse_canary_images(framework, region):
    """
    Return which canary images to run canary tests on for a given framework and AWS region

    :param framework: ML framework (mxnet, tensorflow, pytorch)
    :param region: AWS region
    :return: dlc_images string (space separated string of image URIs)
    """
    customer_type = get_customer_type()
    customer_type_tag = f"-{customer_type}" if customer_type else ""

    version_regex = {
        "tensorflow": rf"tf(-sagemaker)?{customer_type_tag}-(\d+.\d+)\S*(py\d+)",
        "mxnet": rf"mx(-sagemaker)?{customer_type_tag}-(\d+.\d+)\S*(py\d+)",
        "pytorch": rf"pt(-sagemaker)?{customer_type_tag}-(\d+.\d+)\S*(py\d+)",
        "huggingface_pytorch": r"hf-\S*pt(-sagemaker)?-(\d+.\d+)\S*(py\d+)",
        "huggingface_tensorflow": r"hf-\S*tf(-sagemaker)?-(\d+.\d+)\S*(py\d+)",
        "autogluon": r"ag(-sagemaker)?-(\d+.\d+)\S*(py\d+)",
    }

    repo = git.Repo(os.getcwd(), search_parent_directories=True)

    versions_counter = {}
    pre_populated_py3_version = {}

    for tag in repo.tags:
        tag_str = str(tag)
        match = re.search(version_regex[framework], tag_str)
        ## The tags not have -py3 will not pass th condition below
        ## This eliminates all the old and testing tags that we are not monitoring.
        if match:
            version = match.group(2)
            if not versions_counter.get(version):
                versions_counter[version] = {"tr": False, "inf": False}
            if "tr" not in tag_str and "inf" not in tag_str:
                versions_counter[version]["tr"] = True
                versions_counter[version]["inf"] = True
            elif "tr" in tag_str:
                versions_counter[version]["tr"] = True
            elif "inf" in tag_str:
                versions_counter[version]["inf"] = True
            
            if version not in pre_populated_py3_version:
                pre_populated_py3_version[version] = set()
            pre_populated_py3_version[version].add(match.group(3))

    versions = []
    for v, inf_train in versions_counter.items():
        # Earlier versions of huggingface did not have inference
        if (inf_train["inf"] and inf_train["tr"]) or framework.startswith("huggingface"):
            versions.append(v)

    # Sort ascending to descending, use lambda to ensure 2.2 < 2.15, for instance
    versions.sort(key=lambda version_str: [int(point) for point in version_str.split(".")], reverse=True)

    registry = PUBLIC_DLC_REGISTRY
    framework_versions = versions if len(versions) < 4 else versions[:3]
    dlc_images = []
    for fw_version in framework_versions:
        if fw_version in pre_populated_py3_version:
            py3_versions = pre_populated_py3_version[fw_version]
        else:
            ## Keeping it here for legacy releases. Based on the new implementation
            ## the tags always have py3 version appended to it and hence the 
            ## get_canary_default_tag_py3_version might not be required. 
            py3_versions = [get_canary_default_tag_py3_version(framework, fw_version)]
        
        for py3_version in py3_versions:
            images = {
                "tensorflow": [
                    f"{registry}.dkr.ecr.{region}.amazonaws.com/tensorflow-training:{fw_version}-gpu-{py3_version}",
                    f"{registry}.dkr.ecr.{region}.amazonaws.com/tensorflow-training:{fw_version}-cpu-{py3_version}",
                    f"{registry}.dkr.ecr.{region}.amazonaws.com/tensorflow-inference:{fw_version}-gpu",
                    f"{registry}.dkr.ecr.{region}.amazonaws.com/tensorflow-inference:{fw_version}-cpu",
                ],
                "mxnet": [
                    f"{registry}.dkr.ecr.{region}.amazonaws.com/mxnet-training:{fw_version}-gpu-{py3_version}",
                    f"{registry}.dkr.ecr.{region}.amazonaws.com/mxnet-training:{fw_version}-cpu-{py3_version}",
                    f"{registry}.dkr.ecr.{region}.amazonaws.com/mxnet-inference:{fw_version}-gpu-{py3_version}",
                    f"{registry}.dkr.ecr.{region}.amazonaws.com/mxnet-inference:{fw_version}-cpu-{py3_version}",
                ],
                "pytorch": [
                    f"{registry}.dkr.ecr.{region}.amazonaws.com/pytorch-training:{fw_version}-gpu-{py3_version}",
                    f"{registry}.dkr.ecr.{region}.amazonaws.com/pytorch-training:{fw_version}-cpu-{py3_version}",
                    f"{registry}.dkr.ecr.{region}.amazonaws.com/pytorch-inference:{fw_version}-gpu-{py3_version}",
                    f"{registry}.dkr.ecr.{region}.amazonaws.com/pytorch-inference:{fw_version}-cpu-{py3_version}",
                ],
                # TODO: uncomment once cpu training and inference images become available
                "huggingface_pytorch": [
                    f"{registry}.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-training:{fw_version}-gpu-{py3_version}",
                    # f"{registry}.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-training:{fw_version}-cpu-{py3_version}",
                    f"{registry}.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-inference:{fw_version}-gpu-{py3_version}",
                    f"{registry}.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-inference:{fw_version}-cpu-{py3_version}",
                ],
                "huggingface_tensorflow": [
                    f"{registry}.dkr.ecr.{region}.amazonaws.com/huggingface-tensorflow-training:{fw_version}-gpu-{py3_version}",
                    # f"{registry}.dkr.ecr.{region}.amazonaws.com/huggingface-tensorflow-training:{fw_version}-cpu-{py3_version}",
                    f"{registry}.dkr.ecr.{region}.amazonaws.com/huggingface-tensorflow-inference:{fw_version}-gpu-{py3_version}",
                    f"{registry}.dkr.ecr.{region}.amazonaws.com/huggingface-tensorflow-inference:{fw_version}-cpu-{py3_version}",
                ],
                "autogluon": [
                    f"{registry}.dkr.ecr.{region}.amazonaws.com/autogluon-training:{fw_version}-gpu-{py3_version}",
                    f"{registry}.dkr.ecr.{region}.amazonaws.com/autogluon-training:{fw_version}-cpu-{py3_version}",
                ],
            }
            # E3 Images have an additional "e3" tag to distinguish them from the regular "sagemaker" tag
            if customer_type == "e3":
                dlc_images += [f"{img}-e3" for img in images[framework]]
            else:
                dlc_images += images[framework]

    return " ".join(dlc_images)


def setup_sm_benchmark_tf_train_env(resources_location, setup_tf1_env, setup_tf2_env):
    """
    Create a virtual environment for benchmark tests if it doesn't already exist, and download all necessary scripts

    :param resources_location: <str> directory in which test resources should be placed
    :param setup_tf1_env: <bool> True if tf1 resources need to be setup
    :param setup_tf2_env: <bool> True if tf2 resources need to be setup
    :return: absolute path to the location of the virtual environment
    """
    ctx = Context()

    tf_resource_dir_list = []
    if setup_tf1_env:
        tf_resource_dir_list.append("tensorflow1")
    if setup_tf2_env:
        tf_resource_dir_list.append("tensorflow2")

    for resource_dir in tf_resource_dir_list:
        with ctx.cd(os.path.join(resources_location, resource_dir)):
            if not os.path.isdir(os.path.join(resources_location, resource_dir, "horovod")):
                # v0.19.4 is the last version for which horovod example tests are py2 compatible
                ctx.run("git clone -b v0.19.4 https://github.com/horovod/horovod.git")
            if not os.path.isdir(os.path.join(resources_location, resource_dir, "deep-learning-models")):
                # We clone branch tf2 for both 1.x and 2.x tests because tf2 branch contains all necessary files
                ctx.run(f"git clone -b tf2 https://github.com/aws-samples/deep-learning-models.git")

    venv_dir = os.path.join(resources_location, "sm_benchmark_venv")
    if not os.path.isdir(venv_dir):
        ctx.run(f"virtualenv {venv_dir}")
        with ctx.prefix(f"source {venv_dir}/bin/activate"):
            ctx.run("pip install 'sagemaker>=2,<3' awscli boto3 botocore six==1.11")

            # SageMaker TF estimator is coded to only accept framework versions up to 2.1.0 as py2 compatible.
            # Fixing this through the following changes:
            estimator_location = ctx.run(
                "echo $(pip3 show sagemaker |grep 'Location' |sed s/'Location: '//g)/sagemaker/tensorflow/estimator.py"
            ).stdout.strip("\n")
            system = ctx.run("uname -s").stdout.strip("\n")
            sed_input_arg = "'' " if system == "Darwin" else ""
            ctx.run(f"sed -i {sed_input_arg}'s/\[2, 1, 0\]/\[2, 1, 1\]/g' {estimator_location}")
    return venv_dir


def setup_sm_benchmark_mx_train_env(resources_location):
    """
    Create a virtual environment for benchmark tests if it doesn't already exist, and download all necessary scripts
    :param resources_location: <str> directory in which test resources should be placed
    :return: absolute path to the location of the virtual environment
    """
    ctx = Context()

    venv_dir = os.path.join(resources_location, "sm_benchmark_venv")
    if not os.path.isdir(venv_dir):
        ctx.run(f"virtualenv {venv_dir}")
        with ctx.prefix(f"source {venv_dir}/bin/activate"):
            ctx.run("pip install sagemaker awscli boto3 botocore")
    return venv_dir


def setup_sm_benchmark_hf_infer_env(resources_location):
    """
    Create a virtual environment for benchmark tests if it doesn't already exist, and download all necessary scripts
    :param resources_location: <str> directory in which test resources should be placed
    :return: absolute path to the location of the virtual environment
    """
    ctx = Context()

    venv_dir = os.path.join(resources_location, "sm_benchmark_hf_venv")
    if not os.path.isdir(venv_dir):
        ctx.run(f"python3 -m virtualenv {venv_dir}")
        with ctx.prefix(f"source {venv_dir}/bin/activate"):
            ctx.run("pip install sagemaker awscli boto3 botocore")
    return venv_dir


def get_account_id_from_image_uri(image_uri):
    """
    Find the account ID where the image is located

    :param image_uri: <str> ECR image URI
    :return: <str> AWS Account ID
    """
    return image_uri.split(".")[0]


def get_region_from_image_uri(image_uri):
    """
    Find the region where the image is located

    :param image_uri: <str> ECR image URI
    :return: <str> AWS Region Name
    """
    region_pattern = r"(us(-gov)?|ap|ca|cn|eu|sa)-(central|(north|south)?(east|west)?)-\d+"
    region_search = re.search(region_pattern, image_uri)
    assert region_search, f"{image_uri} must have region that matches {region_pattern}"
    return region_search.group()


def get_unique_name_from_tag(image_uri):
    """
    Return the unique from the image tag.
    :param image_uri: ECR image URI
    :return: unique name
    """
    return re.sub("[^A-Za-z0-9]+", "", image_uri)


def get_framework_and_version_from_tag(image_uri):
    """
    Return the framework and version from the image tag.

    :param image_uri: ECR image URI
    :return: framework name, framework version
    """
    tested_framework = get_framework_from_image_uri(image_uri)
    allowed_frameworks = (
        "huggingface_tensorflow",
        "huggingface_pytorch",
        "tensorflow",
        "mxnet",
        "pytorch",
        "autogluon",
    )

    if not tested_framework:
        raise RuntimeError(
            f"Cannot find framework in image uri {image_uri} " f"from allowed frameworks {allowed_frameworks}"
        )

    tag_framework_version = re.search(r"(\d+(\.\d+){1,2})", image_uri).groups()[0]

    return tested_framework, tag_framework_version


# for the time being have this static table. Need to figure out a way to get this from
# neuron github once their version manifest file is updated to the latest
# 1.15.2 etc represent the neuron sdk version
# For each of the sdk version we have differen frameworks like pytoch, mxnet etc
# For each of the frameworks it has the framework version mapping to the actual neuron framework version in the container
# If the framework version does not exist then it means it is not supported for that neuron sdk version
NEURON_VERSION_MANIFEST = {
    "1.15.2": {
        "pytorch": {
            "1.5.1": "1.5.1.1.5.21.0",
            "1.6.0": "1.6.0.1.5.21.0",
            "1.7.1": "1.7.1.1.5.21.0",
            "1.8.1": "1.8.1.1.5.21.0",
        },
        "tensorflow": {
            "2.1.4": "2.1.4.1.6.10.0",
            "2.2.3": "2.2.3.1.6.10.0",
            "2.3.3": "2.3.3.1.6.10.0",
            "2.4.2": "2.4.2.1.6.10.0",
            "2.4.2": "2.4.2.1.6.10.0",
            "2.5.0": "2.5.0.1.6.10.0",
        },
        "mxnet": {
            "1.8.0": "1.8.0.1.3.4.0",
        },
    },
    "1.16.0": {
        "pytorch": {
            "1.5.1": "1.5.1.2.0.318.0",
            "1.7.1": "1.7.1.2.0.318.0",
            "1.8.1": "1.8.1.2.0.318.0",
            "1.9.1": "1.9.1.2.0.318.0",
        },
        "tensorflow": {
            "2.1.4": "2.1.4.2.0.3.0",
            "2.2.3": "2.2.3.2.0.3.0",
            "2.3.4": "2.3.4.2.0.3.0",
            "2.4.3": "2.4.3.2.0.3.0",
            "2.5.1": "2.5.1.2.0.3.0",
            "1.15.5": "1.15.5.2.0.3.0",
        },
        "mxnet": {
            "1.8.0": "1.8.0.2.0.271.0",
        },
    },
    "1.16.1": {
        "pytorch": {
            "1.5.1": "1.5.1.2.0.392.0",
            "1.7.1": "1.7.1.2.0.392.0",
            "1.8.1": "1.8.1.2.0.392.0",
            "1.9.1": "1.9.1.2.0.392.0",
        },
        "tensorflow": {
            "2.1.4": "2.1.4.2.0.4.0",
            "2.2.3": "2.2.3.2.0.4.0",
            "2.3.4": "2.3.4.2.0.4.0",
            "2.4.3": "2.4.3.2.0.4.0",
            "2.5.1": "2.5.1.2.0.4.0",
            "1.15.5": "1.15.5.2.0.4.0",
        },
        "mxnet": {
            "1.8.0": "1.8.0.2.0.276.0",
        },
    },
    "1.17.0": {
        "pytorch": {
            "1.5.1": "1.5.1.2.1.7.0",
            "1.7.1": "1.7.1.2.1.7.0",
            "1.8.1": "1.8.1.2.1.7.0",
            "1.9.1": "1.9.1.2.1.7.0",
            "1.10.1": "1.10.1.2.1.7.0",
        },
        "tensorflow": {
            "2.1.4": "2.1.4.2.0.4.0",
            "2.2.3": "2.2.3.2.0.4.0",
            "2.3.4": "2.3.4.2.0.4.0",
            "2.4.3": "2.4.3.2.0.4.0",
            "2.5.2": "2.5.2.2.1.6.0",
            "1.15.5": "1.15.5.2.1.6.0",
        },
        "mxnet": {
            "1.8.0": "1.8.0.2.1.5.0",
        },
    },
    "1.17.1": {
        "pytorch": {
            "1.10.1": "1.10.1.2.1.7.0",
        },
        "tensorflow": {
            "2.5.2": "2.5.2.2.1.13.0",
            "1.15.5": "1.15.5.2.1.13.0",
        },
        "mxnet": {
            "1.8.0": "1.8.0.2.1.5.0",
        },
    },
    "1.18.0": {
        "pytorch": {
            "1.5.1": "1.5.1.2.2.0.0",
            "1.7.1": "1.7.1.2.2.0.0",
            "1.8.1": "1.8.1.2.2.0.0",
            "1.9.1": "1.9.1.2.2.0.0",
            "1.10.2": "1.10.1.2.2.0.0",
        },
        "tensorflow": {
            "2.5.3": "2.5.3.2.2.0.0",
            "2.6.3": "2.6.3.2.2.0.0",
            "2.7.1": "2.7.1.2.2.0.0",
            "1.15.5": "1.15.5.2.2.0.0",
        },
        "mxnet": {
            "1.8.0": "1.8.0.2.2.2.0",
        },
    },
    "1.19.0": {
        "pytorch": {
            "1.7.1": "1.7.1.2.3.0.0",
            "1.8.1": "1.8.1.2.3.0.0",
            "1.9.1": "1.9.1.2.3.0.0",
            "1.10.2": "1.10.2.2.3.0.0",
            "1.11.0": "1.11.0.2.3.0.0",
        },
        "tensorflow": {
            "2.5.3": "2.5.3.2.3.0.0",
            "2.6.3": "2.6.3.2.3.0.0",
            "2.7.1": "2.7.1.2.3.0.0",
            "2.8.0": "2.8.0.2.3.0.0",
            "1.15.5": "1.15.5.2.3.0.0",
        },
        "mxnet": {
            "1.8.0": "1.8.0.2.2.2.0",
        },
    },
}


def get_neuron_sdk_version_from_tag(image_uri):
    """
    Return the neuron sdk version from the image tag.
    :param image_uri: ECR image URI
    :return: neuron sdk version
    """
    neuron_sdk_version = None

    if "sdk" in image_uri:
        neuron_sdk_version = re.search(r"sdk([\d\.]+)", image_uri).group(1)

    return neuron_sdk_version


def get_neuron_framework_and_version_from_tag(image_uri):
    """
    Return the framework version and expected framework version for the neuron tag from the image tag.

    :param image_uri: ECR image URI
    :return: framework version, expected framework version from neuron sdk version
    """
    tested_framework, tag_framework_version = get_framework_and_version_from_tag(image_uri)
    neuron_sdk_version = get_neuron_sdk_version_from_tag(image_uri)

    if neuron_sdk_version is None:
        return tag_framework_version, None

    if neuron_sdk_version not in NEURON_VERSION_MANIFEST:
        raise KeyError(f"Cannot find neuron sdk version {neuron_sdk_version} ")

    # Framework name may include huggingface
    if tested_framework.startswith("huggingface_"):
        tested_framework = tested_framework[len("huggingface_") :]

    neuron_framework_versions = NEURON_VERSION_MANIFEST[neuron_sdk_version][tested_framework]
    neuron_tag_framework_version = neuron_framework_versions.get(tag_framework_version)

    return tested_framework, neuron_tag_framework_version


def get_framework_from_image_uri(image_uri):
    return (
        "huggingface_tensorflow"
        if "huggingface-tensorflow" in image_uri
        else "huggingface_pytorch"
        if "huggingface-pytorch" in image_uri
        else "mxnet"
        if "mxnet" in image_uri
        else "pytorch"
        if "pytorch" in image_uri
        else "tensorflow"
        if "tensorflow" in image_uri
        else "autogluon"
        if "autogluon" in image_uri
        else None
    )


def get_all_the_tags_of_an_image_from_ecr(ecr_client, image_uri):
    """
    Uses ecr describe to generate all the tags of an image.

    :param ecr_client: boto3 Client for ECR
    :param image_uri: str Image URI
    :return: list, All the image tags
    """
    account_id = get_account_id_from_image_uri(image_uri)
    image_repo_name, image_tag = get_repository_and_tag_from_image_uri(image_uri)
    response = ecr_client.describe_images(
        registryId=account_id,
        repositoryName=image_repo_name,
        imageIds=[
            {
                'imageTag': image_tag
            },
        ]
    )
    return response['imageDetails'][0]['imageTags']


def get_cuda_version_from_tag(image_uri):
    """
    Return the cuda version from the image tag as cuXXX
    :param image_uri: ECR image URI
    :return: cuda version as cuXXX
    """
    cuda_framework_version = None
    cuda_str = ["cu", "gpu"]
    image_region = get_region_from_image_uri(image_uri)
    ecr_client = boto3.Session(region_name=image_region).client('ecr')
    all_image_tags = get_all_the_tags_of_an_image_from_ecr(ecr_client, image_uri)

    for image_tag in all_image_tags:
        if all(keyword in image_tag for keyword in cuda_str):
            cuda_framework_version = re.search(r"(cu\d+)-", image_tag).groups()[0]
            return cuda_framework_version

    if "gpu" in image_uri:
        raise CudaVersionTagNotFoundException()
    else:
        return None


def get_synapseai_version_from_tag(image_uri):
    """
    Return the synapseai version from the image tag.
    :param image_uri: ECR image URI
    :return: synapseai version
    """
    synapseai_version = None

    synapseai_str = ["synapseai", "hpu"]
    if all(keyword in image_uri for keyword in synapseai_str):
        synapseai_version = re.search(r"synapseai(\d+(\.\d+){2})", image_uri).groups()[0]

    return synapseai_version


def get_job_type_from_image(image_uri):
    """
    Return the Job type from the image tag.

    :param image_uri: ECR image URI
    :return: Job Type
    """
    tested_job_type = None
    allowed_job_types = ("training", "inference")
    for job_type in allowed_job_types:
        if job_type in image_uri:
            tested_job_type = job_type
            break

    if not tested_job_type and "eia" in image_uri:
        tested_job_type = "inference"

    if not tested_job_type:
        raise RuntimeError(
            f"Cannot find Job Type in image uri {image_uri} " f"from allowed frameworks {allowed_job_types}"
        )

    return tested_job_type


def get_repository_and_tag_from_image_uri(image_uri):
    """
    Return the name of the repository holding the image

    :param image_uri: URI of the image
    :return: <str> repository name
    """
    repository_uri, tag = image_uri.split(":")
    _, repository_name = repository_uri.split("/")
    return repository_name, tag


def get_processor_from_image_uri(image_uri):
    """
    Return processor from the image URI

    Assumes image uri includes -<processor> in it's tag, where <processor> is one of cpu, gpu or eia.

    :param image_uri: ECR image URI
    :return: cpu, gpu, eia, neuron or hpu
    """
    allowed_processors = ["eia", "neuron", "cpu", "gpu", "hpu"]

    for processor in allowed_processors:
        match = re.search(rf"-({processor})", image_uri)
        if match:
            return match.group(1)
    raise RuntimeError("Cannot find processor")


def get_python_version_from_image_uri(image_uri):
    """
    Return the python version from the image URI

    :param image_uri: ECR image URI
    :return: str py36, py37, py38, etc., based information available in image URI
    """
    python_version_search = re.search(r"py\d+", image_uri)
    if not python_version_search:
        raise MissingPythonVersionException(f"{image_uri} does not have python version in the form 'py\\d+'")
    python_version = python_version_search.group()
    return "py36" if python_version == "py3" else python_version


def get_container_name(prefix, image_uri):
    """
    Create a unique container name based off of a test related prefix and the image uri
    :param prefix: test related prefix, like "emacs" or "pip-check"
    :param image_uri: ECR image URI
    :return: container name
    """
    return f"{prefix}-{image_uri.split('/')[-1].replace('.', '-').replace(':', '-')}"


def stop_and_remove_container(container_name, context):
    """
    Helper function to stop a container locally
    :param container_name: Name of the docker container
    :param context: Invoke context object
    """
    context.run(
        f"docker rm -f {container_name}",
        hide=True,
    )


def start_container(container_name, image_uri, context):
    """
    Helper function to start a container locally
    :param container_name: Name of the docker container
    :param image_uri: ECR image URI
    :param context: Invoke context object
    """
    context.run(
        f"docker run --entrypoint='/bin/bash' --name {container_name} -itd {image_uri}",
        hide=True,
    )


def run_cmd_on_container(container_name, context, cmd, executable="bash", warn=False):
    """
    Helper function to run commands on a locally running container
    :param container_name: Name of the docker container
    :param context: ECR image URI
    :param cmd: Command to run on the container
    :param executable: Executable to run on the container (bash or python)
    :param warn: Whether to only warn as opposed to exit if command fails
    :return: invoke output, can be used to parse stdout, etc
    """
    if executable not in ("bash", "python"):
        LOGGER.warning(f"Unrecognized executable {executable}. It will be run as {executable} -c '{cmd}'")
    return context.run(
        f"docker exec --user root {container_name} {executable} -c '{cmd}'", hide=True, warn=warn, timeout=60
    )

def uniquify_list_of_dict(list_of_dict):
    """
    Takes list_of_dict as an input and returns a list of dict such that each dict is only present
    once in the returned list. Runs an operation that is similar to list(set(input_list)). However,
    for list_of_dict, it is not possible to run the operation directly. 

    :param list_of_dict: List(dict)
    :return: List(dict)
    """
    list_of_string = [json.dumps(dict_element, sort_keys=True) for dict_element in list_of_dict]
    unique_list_of_string = list(set(list_of_string))
    list_of_dict_to_return = [json.loads(str_element) for str_element in unique_list_of_string]
    return list_of_dict_to_return

def get_tensorflow_model_base_path(image_uri):
    """
    Retrieve model base path based on version of TensorFlow
    Requirement: Model defined in TENSORFLOW_MODELS_PATH should be hosted in S3 location for TF version less than 2.6.
                 Starting TF2.7, the models are referred locally as the support for S3 is moved to a separate python package `tensorflow-io`
    :param image_uri: ECR image URI
    :return: <string> model base path
    """
    if is_below_framework_version("2.7", image_uri, "tensorflow"):
        model_base_path = TENSORFLOW_MODELS_BUCKET
    else:
        model_base_path = f"/tensorflow_model/"
    return model_base_path


def build_tensorflow_inference_command_tf27_and_above(model_name):
    """
    Construct the command to download tensorflow model from S3 and start tensorflow model server
    :param model_name:
    :return: <list> command to send to the container
    """
    inference_command = f"mkdir -p /tensorflow_model && aws s3 sync {TENSORFLOW_MODELS_BUCKET}/{model_name}/ /tensorflow_model/{model_name} && /usr/bin/tf_serving_entrypoint.sh"
    return inference_command


def get_tensorflow_inference_environment_variables(model_name, model_base_path):
    """
    Get method for environment variables for tensorflow inference for EC2 and ECS
    :param model_name:
    :return: <list> JSON
    """
    tensorflow_inference_environment_variables = [
        {"name": "MODEL_NAME", "value": model_name},
        {"name": "MODEL_BASE_PATH", "value": model_base_path},
    ]

    return tensorflow_inference_environment_variables


def get_eks_k8s_test_type_label(image_uri):
    """
    Get node label required for k8s job to be scheduled on compatible EKS node
    :param image_uri: ECR image URI
    :return: <string> node label
    """
    if "graviton" in image_uri:
        test_type = "graviton"
    elif "neuron" in image_uri:
        test_type = "neuron"
    else:
        test_type = "gpu"
    return test_type


def execute_env_variables_test(image_uri, env_vars_to_test, container_name_prefix):
    """
    Based on a dictionary of ENV_VAR: val, test that the enviornment variables are correctly set in the container.

    @param image_uri: ECR image URI
    @param env_vars_to_test: dict {"ENV_VAR": "env_var_expected_value"}
    @param container_name_prefix: container name prefix describing test
    """
    ctx = Context()
    container_name = get_container_name(container_name_prefix, image_uri)

    start_container(container_name, image_uri, ctx)
    for var, expected_val in env_vars_to_test.items():
        output = run_cmd_on_container(container_name, ctx, f"echo ${var}")
        actual_val = output.stdout.strip()
        if actual_val:
            assertion_error_sentence = f"It is currently set to {actual_val}."
        else:
            assertion_error_sentence = "It is currently not set."
        assert actual_val == expected_val, \
            f"Environment variable {var} is expected to be {expected_val}. {assertion_error_sentence}."
    stop_and_remove_container(container_name, ctx)


def is_image_available_locally(image_uri):
    """
    Check if the image exists locally.

    :param image_uri: str, image that needs to be checked
    :return: bool, True if image exists locally, otherwise false
    """
    run_output = run(f"docker inspect {image_uri}", hide=True, warn=True)
    return run_output.ok
