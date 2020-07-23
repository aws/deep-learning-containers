import os
import re
from time import sleep

import pytest

from test import test_utils
from test.test_utils.ec2 import get_ec2_instance_type
from test.dlc_tests.conftest import LOGGER

TENSORFLOW1_VERSION = "1."
TENSORFLOW2_VERSION = "2."


TF_EC2_GPU_INSTANCE_TYPE = get_ec2_instance_type(default="g3.8xlarge", processor="gpu")
TF_EC2_CPU_INSTANCE_TYPE = get_ec2_instance_type(default="c5.4xlarge", processor="cpu")


@pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_ec2_tenorflow_inference_gpu(tensorflow_inference, ec2_connection, region, gpu_only):
    run_ec2_tensorflow_inference(tensorflow_inference, ec2_connection, "8500", region)


@pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_ec2_tensorflow_inference_cpu(tensorflow_inference, ec2_connection, region, cpu_only):
    run_ec2_tensorflow_inference(tensorflow_inference, ec2_connection, "8500", region)


@pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
@pytest.mark.parametrize("ei_accelerator_type", ["eia1.large"], indirect=True)
def test_ec2_tensorflow_inference_eia_cpu(tensorflow_inference_eia, ec2_connection, region, eia_only):
    run_ec2_tensorflow_inference(tensorflow_inference_eia, ec2_connection, "8500", region)


@pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
@pytest.mark.parametrize("ei_accelerator_type", ["eia1.large"], indirect=True)
def test_ec2_tensorflow_inference_eia_gpu(tensorflow_inference_eia, ec2_connection, region, eia_only):
    run_ec2_tensorflow_inference(tensorflow_inference_eia, ec2_connection, "8500", region)

def run_ec2_tensorflow_inference(image_uri, ec2_connection, grpc_port, region):
    repo_name, image_tag = image_uri.split("/")[-1].split(":")
    container_name = f"{repo_name}-{image_tag}-ec2"
    framework_version = get_tensorflow_framework_version(image_uri)
    home_dir = ec2_connection.run("echo $HOME").stdout.strip('\n')
    serving_folder_path = os.path.join(home_dir, "serving")
    model_path = os.path.join(serving_folder_path, "models", "mnist")
    mnist_client_path = os.path.join(
        serving_folder_path, "tensorflow_serving", "example", "mnist_client.py"
    )
    docker_cmd = "nvidia-docker" if "gpu" in image_uri else "docker"
    docker_run_cmd = (
        f"{docker_cmd} run -id --name {container_name} -p {grpc_port}:8500 "
        f"--mount type=bind,source={model_path},target=/models/mnist -e MODEL_NAME=mnist"
        f" {image_uri}"
    )
    try:
        host_setup_for_tensorflow_inference(
            serving_folder_path, framework_version, ec2_connection
        )
        sleep(2)
        train_mnist_model(serving_folder_path, ec2_connection)
        sleep(10)
        ec2_connection.run(
            f"$(aws ecr get-login --no-include-email --region {region})", hide=True
        )
        LOGGER.info(docker_run_cmd)
        ec2_connection.run(docker_run_cmd, hide=True)
        sleep(20)
        test_utils.request_tensorflow_inference_grpc(
            script_file_path=mnist_client_path, port=grpc_port, connection=ec2_connection
        )
    finally:
        ec2_connection.run(f"docker rm -f {container_name}", warn=True, hide=True)


def get_tensorflow_framework_version(image_uri):
    return re.findall(r"[1-2]\.[0-9][\d|\.]+", image_uri)[0]


def train_mnist_model(serving_folder_path, ec2_connection):
    ec2_connection.run(f"cd {serving_folder_path}")
    mnist_script_path = f"{serving_folder_path}/tensorflow_serving/example/mnist_saved_model.py"
    ec2_connection.run(
        f"python {mnist_script_path} {serving_folder_path}/models/mnist", hide=True
    )


def host_setup_for_tensorflow_inference(serving_folder_path, framework_version, ec2_connection):
    # Tensorflow 1.x doesn't have package with version 1.15.2 so use only 1.15
    ec2_connection.run(
        (
            f"pip install --user -qq -U 'tensorflow<={framework_version}' "
            f" 'tensorflow-serving-api<={framework_version}'"
        ), hide=True
    )
    if os.path.exists(f"{serving_folder_path}"):
        ec2_connection.run(f"rm -rf {serving_folder_path}")
    if str(framework_version).startswith(TENSORFLOW1_VERSION):
        run_out = ec2_connection.run(
            f"git clone https://github.com/tensorflow/serving.git {serving_folder_path}"
        )
        git_branch_version = re.findall(r"[1-2]\.[0-9]\d", framework_version)[0]
        ec2_connection.run(
            f"cd {serving_folder_path} && git checkout r{git_branch_version}"
        )
        LOGGER.info(f"Clone TF serving repository status {run_out.return_code == 0}")
    else:
        local_scripts_path = os.path.join("container_tests", "bin", "tensorflow_serving")
        ec2_connection.run(f"mkdir -p {serving_folder_path}")
        ec2_connection.run(f"cp -r {local_scripts_path} {serving_folder_path}")
