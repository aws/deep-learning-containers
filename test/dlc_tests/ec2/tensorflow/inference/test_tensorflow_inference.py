import os
import re
from time import sleep

import pytest

from test import test_utils
from test.test_utils.ec2 import get_ec2_instance_type, get_ec2_accelerator_type
from test.dlc_tests.conftest import LOGGER

TENSORFLOW1_VERSION = "1."
TENSORFLOW2_VERSION = "2."


TF_EC2_GPU_INSTANCE_TYPE = get_ec2_instance_type(default="g3.8xlarge", processor="gpu")
TF_EC2_CPU_INSTANCE_TYPE = get_ec2_instance_type(default="c5.4xlarge", processor="cpu")
TF_EC2_EIA_ACCELERATOR_TYPE = get_ec2_accelerator_type(default="eia1.large", processor="eia")
TF_EC2_NEURON_ACCELERATOR_TYPE = get_ec2_accelerator_type(default="inf1.xlarge", processor="neuron")

@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_NEURON_ACCELERATOR_TYPE, indirect=True)
#FIX ME: Sharing the AMI from neuron account to DLC account; use public DLAMI with inf1 support instead
@pytest.mark.parametrize("ec2_instance_ami", [test_utils.NEURON_AL2_DLAMI], indirect=True)
def test_ec2_tensorflow_inference_neuron(tensorflow_inference_neuron, ec2_connection, region, neuron_only):
    run_ec2_tensorflow_inference(tensorflow_inference_neuron, ec2_connection, "8500", region)

@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_ec2_tensorflow_inference_gpu(tensorflow_inference, ec2_connection, region, gpu_only):
    run_ec2_tensorflow_inference(tensorflow_inference, ec2_connection, "8500", region)


@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_ec2_tensorflow_inference_cpu(tensorflow_inference, ec2_connection, region, cpu_only):
    run_ec2_tensorflow_inference(tensorflow_inference, ec2_connection, "8500", region)


@pytest.mark.integration("elastic_inference")
@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
@pytest.mark.parametrize("ei_accelerator_type", TF_EC2_EIA_ACCELERATOR_TYPE, indirect=True)
def test_ec2_tensorflow_inference_eia_cpu(tensorflow_inference_eia, ec2_connection, region, eia_only):
    run_ec2_tensorflow_inference(tensorflow_inference_eia, ec2_connection, "8500", region)


@pytest.mark.integration("elastic_inference")
@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
@pytest.mark.parametrize("ei_accelerator_type", TF_EC2_EIA_ACCELERATOR_TYPE, indirect=True)
def test_ec2_tensorflow_inference_eia_gpu(tensorflow_inference_eia, ec2_connection, region, eia_only):
    run_ec2_tensorflow_inference(tensorflow_inference_eia, ec2_connection, "8500", region)


@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", ["p2.xlarge"], indirect=True)
def test_ec2_tensorflow_inference_gpu_telemetry(tensorflow_inference, ec2_connection, region, gpu_only):
    run_ec2_tensorflow_inference(tensorflow_inference, ec2_connection, "8500", region, True)


@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", ["c5.4xlarge"], indirect=True)
def test_ec2_tensorflow_inference_cpu_telemetry(tensorflow_inference, ec2_connection, region, cpu_only):
    run_ec2_tensorflow_inference(tensorflow_inference, ec2_connection, "8500", region, True)


def run_ec2_tensorflow_inference(image_uri, ec2_connection, grpc_port, region, telemetry_mode=False):
    repo_name, image_tag = image_uri.split("/")[-1].split(":")
    container_name = f"{repo_name}-{image_tag}-ec2"
    framework_version = get_tensorflow_framework_version(image_uri)
    home_dir = ec2_connection.run("echo $HOME").stdout.strip('\n')
    serving_folder_path = os.path.join(home_dir, "serving")
    model_path = os.path.join(serving_folder_path, "models", "mnist")
    mnist_client_path = os.path.join(
        serving_folder_path, "tensorflow_serving", "example", "mnist_client.py"
    )
    
    is_neuron = "neuron" in image_uri
         

    docker_cmd = "nvidia-docker" if "gpu" in image_uri else "docker"
    docker_run_cmd = ""
    if is_neuron:
        docker_run_cmd = (
            f"{docker_cmd} run -id --name {container_name} -p {grpc_port}:8500 "
            f"--device=/dev/neuron0 --net=host  --cap-add IPC_LOCK "
            f"--mount type=bind,source={model_path},target=/models/mnist -e TEST_MODE=1 -e MODEL_NAME=mnist"
            f" {image_uri}"
        )
    else:
        docker_run_cmd = (
            f"{docker_cmd} run -id --name {container_name} -p {grpc_port}:8500 "
            f"--mount type=bind,source={model_path},target=/models/mnist -e TEST_MODE=1 -e MODEL_NAME=mnist"
            f" {image_uri}"
        )
    try:
        host_setup_for_tensorflow_inference(
            serving_folder_path, framework_version, ec2_connection, is_neuron, 'mnist'
        )
        sleep(2)
        if not is_neuron:
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
        if telemetry_mode:
            check_telemetry(ec2_connection, container_name)
    finally:
        ec2_connection.run(f"docker rm -f {container_name}", warn=True, hide=True)


def get_tensorflow_framework_version(image_uri):
    return re.findall(r"[1-2]\.[0-9][\d|\.]+", image_uri)[0]


def train_mnist_model(serving_folder_path, ec2_connection):
    ec2_connection.run(f"cd {serving_folder_path}")
    mnist_script_path = f"{serving_folder_path}/tensorflow_serving/example/mnist_saved_model.py"
    ec2_connection.run(
        f"python3 {mnist_script_path} {serving_folder_path}/models/mnist", hide=True
    )


def host_setup_for_tensorflow_inference(serving_folder_path, framework_version, ec2_connection, is_neuron, model_name):
    # Tensorflow 1.x doesn't have package with version 1.15.2 so use only 1.15
    ec2_connection.run(
        (
            f"pip3 install --user -qq -U 'tensorflow<={framework_version}' "
            f" 'tensorflow-serving-api<={framework_version}' "
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
        if is_neuron:
            container_test_local_file = os.path.join("$HOME", "container_tests/bin/neuron_tests/mnist_client.py")
            ec2_connection.run(f"cp -f {container_test_local_file} {serving_folder_path}/tensorflow_serving/example")
            neuron_model_file_path = os.path.join(serving_folder_path, f"models/{model_name}/1")
            neuron_model_file = os.path.join(neuron_model_file_path, "saved_model.pb")
            LOGGER.info(f"Host Model path {neuron_model_file_path}")
            ec2_connection.run(f"mkdir -p {neuron_model_file_path}")
            model_file_path = f"https://aws-dlc-sample-models.s3.amazonaws.com/{model_name}_neuron/1/saved_model.pb"
            model_dwld = (
                f"wget -O {neuron_model_file} {model_file_path} "
            )
            ec2_connection.run(model_dwld)
    else:
        local_scripts_path = os.path.join("container_tests", "bin", "tensorflow_serving")
        ec2_connection.run(f"mkdir -p {serving_folder_path}")
        ec2_connection.run(f"cp -r {local_scripts_path} {serving_folder_path}")


def check_telemetry(ec2_connection, container_name):
    telemetry_cmd = "import os; assert (os.path.exists('/tmp/test_request.txt'))"
    ec2_connection.run(
        f'''docker exec -it {container_name} python -c {telemetry_cmd} ''',
        hide=True, warn=True
    )
