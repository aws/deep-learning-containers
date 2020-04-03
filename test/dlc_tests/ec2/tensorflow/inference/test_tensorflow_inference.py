import os
from time import sleep

import pytest

from test import test_utils
from test.dlc_tests.conftest import LOGGER

TENSORFLOW1_VERSION = "1."
TENSORFLOW2_VERSION = "2."


@pytest.mark.parametrize("ec2_instance_type", ["p3.2xlarge"], indirect=True)
def test_ec2_tenorflow_inference_gpu(tensorflow_inference, ec2_connection, region, gpu_only):
    run_ec2_tensorflow_inference(tensorflow_inference, ec2_connection, "8500", region)


@pytest.mark.parametrize("ec2_instance_type", ["c5.4xlarge"], indirect=True)
def test_ec2_tensorflow_inference_cpu(tensorflow_inference, ec2_connection, region, cpu_only):
    run_ec2_tensorflow_inference(tensorflow_inference, ec2_connection, "8500", region)


def run_ec2_tensorflow_inference(image_uri, ec2_connection, grpc_port, region):
    repo_name, image_tag = image_uri.split("/")[-1].split(":")
    container_name = f"{repo_name}-{image_tag}-ec2"
    framework_version = get_tensorflow_framework_version(image_uri)
    serving_folder_path = f"{test_utils.UBUNTU_HOME_DIR}/serving"
    model_path = os.path.join(serving_folder_path, "models", "mnist")
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
        sleep(2)
        ec2_connection.run(
            f"$(aws ecr get-login --no-include-email --region {region})", hide=True
        )
        LOGGER.info(docker_run_cmd)
        ec2_connection.run(docker_run_cmd, hide=True)
        inference_result = test_utils.request_tensorflow_inference(
            port=grpc_port, connection=ec2_connection
        )
        sleep(15)
        assert (
            inference_result
        ), f"Failed to perform tensorflow inference test for image: {image_uri} on ec2"

    finally:
        ec2_connection.run(f"docker rm -f {container_name}", warn=True, hide=True)


def get_tensorflow_framework_version(image_uri):
    return image_uri.split("-")[-1]


def train_mnist_model(serving_folder_path, ec2_connection):
    ec2_connection.run(f"cd {serving_folder_path}")
    run_out = ec2_connection.run(
        "python tensorflow_serving/example/mnist_saved_model.py models/mnist", hide=True
    )
    LOGGER.info(
        "Train TF Mnist model for inference.", test_status=(run_out.return_code == 0)
    )
    return run_out.return_code == 0


def host_setup_for_tensorflow_inference(serving_folder_path, framework_version, ec2_connection):
    # Tensorflow 1.x doesn't have package with version 1.15.2 so use only 1.15
    if TENSORFLOW1_VERSION in framework_version:
        framework_version = framework_version[:4]
    run_out = ec2_connection.run(
        (
            f"pip install --user -U tensorflow=={framework_version} "
            f"tensorflow-serving-api=={framework_version}"
        )
    )
    LOGGER.info(
        f"Install pip package for tensorflow inference status : {run_out.return_code == 0}"
    )
    if TENSORFLOW1_VERSION == framework_version[:2]:
        if os.path.exists(f"{serving_folder_path}"):
            ec2_connection.run(f"rm -rf {serving_folder_path}")
        run_out = ec2_connection.run(
            "git clone https://github.com/tensorflow/serving.git"
        )
        ec2_connection.run(
            f"cd {serving_folder_path} && git checkout r{framework_version[:4]}"
        )
        LOGGER.info(f"Clone TF serving repository status {run_out.return_code == 0}")
    else:
        local_scripts_path = os.path.join(
            "container_tests", "bin", "tensorflow_serving", "example"
        )
        ec2_connection.run(f"cp -r {local_scripts_path} {serving_folder_path}")
        training_script = os.path.join(serving_folder_path, "mnist_saved_model.py")
        model_path = os.path.join(serving_folder_path, "models", "mnist")
        ec2_connection.run(f"python {training_script} {model_path}", hide=True)
    return run_out.return_code == 0
