import json
import os
import re
import subprocess
import time
import logging
import sys

import pytest
import boto3

from botocore.exceptions import ClientError
from invoke import run
from invoke.context import Context
from retrying import retry

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stderr))

# Constant to represent default region for boto3 commands
DEFAULT_REGION = "us-west-2"
# Constant to represent AMI Id used to spin up EC2 instances
UBUNTU_16_BASE_DLAMI = "ami-0e57002aaafd42113"
ECS_AML2_GPU_USWEST2 = "ami-09ef8c43fa060063d"
ECS_AML2_CPU_USWEST2 = "ami-014a2e30da708ee8b"

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


def is_tf1(image_uri):
    if "tensorflow" not in image_uri:
        return False
    return bool(re.search(r'1\.\d+\.\d+', image_uri))


def is_tf2(image_uri):
    if "tensorflow" not in image_uri:
        return False
    return bool(re.search(r'2\.\d+\.\d+', image_uri))


def is_pr_context():
    return os.getenv("BUILD_CONTEXT") == "PR"


def is_canary_context():
    return os.getenv("BUILD_CONTEXT") == "CANARY"


def run_subprocess_cmd(cmd, failure="Command failed"):
    command = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
    if command.returncode:
        pytest.fail(f"{failure}. Error log:\n{command.stdout.decode()}")
    return command


def retry_if_result_is_false(result):
    """Return True if we should retry (in this case retry if the result is False), False otherwise"""
    return result is False


@retry(
    stop_max_attempt_number=10,
    wait_fixed=10000,
    retry_on_result=retry_if_result_is_false,
)
def request_mxnet_inference_squeezenet(ip_address="127.0.0.1", port="80", connection=None):
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

    run_out = conn_run(f"curl -X POST http://{ip_address}:{port}/predictions/squeezenet -T kitten.jpg", warn=True)

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
        (f"curl -X POST http://{ip_address}:{port}/predictions/bert_sst/predict -F "
         "'data=[\"Positive sentiment\", \"Negative sentiment\"]'"),
        warn=True
    )

    # The run_out.return_code is not reliable, since sometimes predict request may succeed but the returned result
    # is 404. Hence the extra check.
    if run_out.return_code != 0 or '1' not in run_out.stdout:
        return False

    return True


@retry(
    stop_max_attempt_number=10,
    wait_fixed=10000,
    retry_on_result=retry_if_result_is_false,
)
def request_pytorch_inference_densenet(ip_address="127.0.0.1", port="80", connection=None):
    """
    Send request to container to test inference on flower.jpg
    :param ip_address: str
    :param port: str
    :param connection: obj
    :return: <bool> True/False based on result of inference
    """
    conn_run = connection.run if connection is not None else run
    # Check if image already exists
    run_out = conn_run("[ -f flower.jpg ]", warn=True)
    if run_out.return_code != 0:
        conn_run("curl -O https://s3.amazonaws.com/model-server/inputs/flower.jpg", hide=True)

    run_out = conn_run(f"curl -X POST http://{ip_address}:{port}/predictions/pytorch-densenet -T flower.jpg", hide=True, warn=True)

    # The run_out.return_code is not reliable, since sometimes predict request may succeed but the returned result
    # is 404. Hence the extra check.
    if run_out.return_code != 0 or 'flowerpot' not in run_out.stdout:
        return False

    return True


@retry(stop_max_attempt_number=20, wait_fixed=10000, retry_on_result=retry_if_result_is_false)
def request_tensorflow_inference(model_name, ip_address="127.0.0.1", port="8501"):
    """
    Method to run tensorflow inference on half_plus_two model using CURL command
    :param model_name:
    :param ip_address:
    :param port:
    :connection: ec2_connection object to run the commands remotely over ssh
    :return:
    """
    inference_string = "'{\"instances\": [1.0, 2.0, 5.0]}'"
    run_out = run(
        f"curl -d {inference_string} -X POST  http://{ip_address}:{port}/v1/models/{model_name}:predict", warn=True
    )

    # The run_out.return_code is not reliable, since sometimes predict request may succeed but the returned result
    # is 404. Hence the extra check.
    if run_out.return_code != 0 or 'predictions' not in run_out.stdout:
        return False

    return True


def request_tensorflow_inference_grpc(script_file_path, ip_address="127.0.0.1", port="8500", connection=None):
    """
    Method to run tensorflow inference on MNIST model using gRPC protocol
    :param script_file_path:
    :param ip_address:
    :param port:
    :param connection:
    :return:
    """
    conn_run = connection.run if connection is not None else run
    conn_run(f"python {script_file_path} --num_tests=1000 --server={ip_address}:{port}", hide=True)


def get_mms_run_command(model_names, processor="cpu"):
    """
    Helper function to format run command for MMS
    :param model_names:
    :param processor:
    :return: <str> Command to start MMS server with given model
    """
    if processor != "eia":
        mxnet_model_location = {
            "squeezenet": "https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model",
            "pytorch-densenet": "https://dlc-samples.s3.amazonaws.com/pytorch/multi-model-server/densenet/densenet.mar",
            "bert_sst": "https://aws-dlc-sample-models.s3.amazonaws.com/bert_sst/bert_sst.mar"
        }
    else:
        mxnet_model_location = {
            "resnet-152-eia": "https://s3.amazonaws.com/model-server/model_archive_1.0/resnet-152-eia.mar"
        }

    if not isinstance(model_names, list):
        model_names = [model_names]

    for model_name in model_names:
        if model_name not in mxnet_model_location:
            raise Exception(
                "No entry found for model {} in dictionary".format(model_name)
            )

    parameters = [
        "{}={}".format(name, mxnet_model_location[name]) for name in model_names
    ]
    mms_command = (
        "mxnet-model-server --start --mms-config /home/model-server/config.properties --models "
        + " ".join(parameters)
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
        "saved_model_half_plus_three": {"eia": "saved_model_half_plus_three"},
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
        raise ClientError(e)

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
    if is_pr_context():
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


def parse_canary_images(framework, region):
    tf1 = "1.15"
    tf2 = "2.2"
    mx = "1.6"
    pt = "1.5"

    if framework == "tensorflow":
        framework = "tensorflow2" if "tensorflow2" in os.getenv("CODEBUILD_BUILD_ID") else "tensorflow1"

    images = {
        "tensorflow1":
            f"763104351884.dkr.ecr.{region}.amazonaws.com/tensorflow-training:{tf1}-gpu-py3 "
            f"763104351884.dkr.ecr.{region}.amazonaws.com/tensorflow-training:{tf1}-cpu-py3 "
            f"763104351884.dkr.ecr.{region}.amazonaws.com/tensorflow-training:{tf1}-cpu-py2 "
            f"763104351884.dkr.ecr.{region}.amazonaws.com/tensorflow-training:{tf1}-gpu-py2 "
            f"763104351884.dkr.ecr.{region}.amazonaws.com/tensorflow-inference:{tf1}-gpu "
            f"763104351884.dkr.ecr.{region}.amazonaws.com/tensorflow-inference:{tf1}-cpu",
        "tensorflow2":
            f"763104351884.dkr.ecr.{region}.amazonaws.com/tensorflow-training:{tf2}-gpu-py37 "
            f"763104351884.dkr.ecr.{region}.amazonaws.com/tensorflow-training:{tf2}-cpu-py37",
        "mxnet":
            f"763104351884.dkr.ecr.{region}.amazonaws.com/mxnet-training:{mx}-gpu-py3 "
            f"763104351884.dkr.ecr.{region}.amazonaws.com/mxnet-training:{mx}-cpu-py3 "
            f"763104351884.dkr.ecr.{region}.amazonaws.com/mxnet-training:{mx}-gpu-py2 "
            f"763104351884.dkr.ecr.{region}.amazonaws.com/mxnet-training:{mx}-cpu-py2 "
            f"763104351884.dkr.ecr.{region}.amazonaws.com/mxnet-inference:{mx}-gpu-py3 "
            f"763104351884.dkr.ecr.{region}.amazonaws.com/mxnet-inference:{mx}-cpu-py3 "
            f"763104351884.dkr.ecr.{region}.amazonaws.com/mxnet-inference:{mx}-gpu-py2 "
            f"763104351884.dkr.ecr.{region}.amazonaws.com/mxnet-inference:{mx}-cpu-py2",
        "pytorch":
            f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:{pt}-gpu-py3 "
            f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:{pt}-cpu-py3 "
            f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-inference:{pt}-gpu-py3 "
            f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-inference:{pt}-cpu-py3"
    }
    return images[framework]


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
                ctx.run("git clone https://github.com/horovod/horovod.git")
            if not os.path.isdir(os.path.join(resources_location, resource_dir, "deep-learning-models")):
                # We clone branch tf2 for both 1.x and 2.x tests because tf2 branch contains all necessary files
                ctx.run(f"git clone -b tf2 https://github.com/aws-samples/deep-learning-models.git")

    venv_dir = os.path.join(resources_location, "sm_benchmark_venv")
    if not os.path.isdir(venv_dir):
        ctx.run(f"virtualenv {venv_dir}")
        with ctx.prefix(f"source {venv_dir}/bin/activate"):
            ctx.run("pip install -U sagemaker awscli boto3 botocore six==1.11")
    return venv_dir
