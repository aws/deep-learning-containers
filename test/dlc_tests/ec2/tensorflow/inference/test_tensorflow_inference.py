import os
import re
from time import sleep

import pytest

import test.test_utils.ec2 as ec2_utils

from test import test_utils
from test.test_utils.ec2 import get_ec2_instance_type, get_ec2_accelerator_type
from test.dlc_tests.conftest import LOGGER

TENSORFLOW1_VERSION = "1."
TENSORFLOW2_VERSION = "2."


TF_EC2_GPU_INSTANCE_TYPE = get_ec2_instance_type(default="g3.8xlarge", processor="gpu")
TF_EC2_CPU_INSTANCE_TYPE = get_ec2_instance_type(default="c5.4xlarge", processor="cpu")
TF_EC2_EIA_ACCELERATOR_TYPE = get_ec2_accelerator_type(default="eia1.large", processor="eia")
TF_EC2_NEURON_ACCELERATOR_TYPE = get_ec2_instance_type(default="inf1.xlarge", processor="neuron")
TF_EC2_SINGLE_GPU_INSTANCE_TYPE = get_ec2_instance_type(
    default="p3.2xlarge", processor="gpu", filter_function=ec2_utils.filter_only_single_gpu,
)
TF_EC2_GRAVITON_INSTANCE_TYPE = get_ec2_instance_type(default="c6g.4xlarge", processor="cpu", arch_type="graviton")


@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_NEURON_ACCELERATOR_TYPE, indirect=True)
#FIX ME: Sharing the AMI from neuron account to DLC account; use public DLAMI with inf1 support instead
@pytest.mark.parametrize("ec2_instance_ami", [test_utils.NEURON_UBUNTU_18_BASE_DLAMI_US_WEST_2], indirect=True)
def test_ec2_tensorflow_inference_neuron(tensorflow_inference_neuron, ec2_connection, ec2_instance_ami, region):
    run_ec2_tensorflow_inference(tensorflow_inference_neuron, ec2_connection, ec2_instance_ami, "8500", region)


@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_ec2_tensorflow_inference_gpu(tensorflow_inference, ec2_connection, ec2_instance_ami, region, gpu_only, ec2_instance_type):
    if test_utils.is_image_incompatible_with_instance_type(tensorflow_inference, ec2_instance_type):
        pytest.skip(f"Image {tensorflow_inference} is incompatible with instance type {ec2_instance_type}")
    run_ec2_tensorflow_inference(tensorflow_inference, ec2_connection, ec2_instance_ami, "8500", region)


@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_ec2_tensorflow_inference_cpu(tensorflow_inference, ec2_connection, ec2_instance_ami, region, cpu_only):
    run_ec2_tensorflow_inference(tensorflow_inference, ec2_connection, ec2_instance_ami, "8500", region)


@pytest.mark.integration("elastic_inference")
@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
@pytest.mark.parametrize("ei_accelerator_type", TF_EC2_EIA_ACCELERATOR_TYPE, indirect=True)
def test_ec2_tensorflow_inference_eia_cpu(tensorflow_inference_eia, ec2_connection, ec2_instance_ami, region):
    run_ec2_tensorflow_inference(tensorflow_inference_eia, ec2_connection, ec2_instance_ami, "8500", region)


@pytest.mark.integration("elastic_inference")
@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
@pytest.mark.parametrize("ei_accelerator_type", TF_EC2_EIA_ACCELERATOR_TYPE, indirect=True)
def test_ec2_tensorflow_inference_eia_gpu(tensorflow_inference_eia, ec2_connection, ec2_instance_ami, region, ec2_instance_type):
    if ec2_instance_type == "p4d.24xlarge":
        pytest.skip(f"Skipping EIA GPU test for {ec2_instance_type} instance type. See https://github.com/aws/deep-learning-containers/issues/962")
    run_ec2_tensorflow_inference(tensorflow_inference_eia, ec2_connection, ec2_instance_ami, "8500", region)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_SINGLE_GPU_INSTANCE_TYPE, indirect=True)
def test_ec2_tensorflow_inference_gpu_telemetry(
        tensorflow_inference, ec2_connection, ec2_instance_ami, region, gpu_only, ec2_instance_type
):
    if test_utils.is_image_incompatible_with_instance_type(tensorflow_inference, ec2_instance_type):
        pytest.skip(f"Image {tensorflow_inference} is incompatible with instance type {ec2_instance_type}")
    run_ec2_tensorflow_inference(tensorflow_inference, ec2_connection, ec2_instance_ami, "8500", region, True)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_ec2_tensorflow_inference_cpu_telemetry(tensorflow_inference, ec2_connection, ec2_instance_ami, region, cpu_only):
    run_ec2_tensorflow_inference(tensorflow_inference, ec2_connection, ec2_instance_ami, "8500", region, True)


@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_GRAVITON_INSTANCE_TYPE, indirect=True)
@pytest.mark.parametrize("ec2_instance_ami", [test_utils.UL18_CPU_ARM64_US_WEST_2], indirect=True)
def test_ec2_tensorflow_inference_graviton_cpu(tensorflow_inference_graviton, ec2_connection, ec2_instance_ami, region, cpu_only):
    run_ec2_tensorflow_inference(tensorflow_inference_graviton, ec2_connection, ec2_instance_ami, "8500", region)

@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_GRAVITON_INSTANCE_TYPE, indirect=True)
@pytest.mark.parametrize("ec2_instance_ami", [test_utils.UL18_CPU_ARM64_US_WEST_2], indirect=True)
def test_ec2_tensorflow_inference_graviton_cpu_telemetry(tensorflow_inference_graviton, ec2_connection, ec2_instance_ami, region, cpu_only):
    run_ec2_tensorflow_inference(tensorflow_inference_graviton, ec2_connection, ec2_instance_ami, "8500", region, True)
    

def run_ec2_tensorflow_inference(image_uri, ec2_connection, ec2_instance_ami, grpc_port, region, telemetry_mode=False):
    repo_name, image_tag = image_uri.split("/")[-1].split(":")
    container_name = f"{repo_name}-{image_tag}-ec2"
    framework_version = get_tensorflow_framework_version(image_uri)
    home_dir = ec2_connection.run("echo $HOME").stdout.strip('\n')
    serving_folder_path = os.path.join(home_dir, "serving")
    model_name = "mnist"
    model_path = os.path.join(serving_folder_path, "models", model_name)
    python_invoker = test_utils.get_python_invoker(ec2_instance_ami)
    mnist_client_path = os.path.join(
        serving_folder_path, "tensorflow_serving", "example", "mnist_client.py"
    )

    is_neuron = "neuron" in image_uri
    is_graviton = "graviton" in image_uri

    docker_cmd = "nvidia-docker" if "gpu" in image_uri else "docker"
    if is_neuron:
        #For 2.5 using rest api port instead of grpc since using curl for prediction instead of grpc
        if str(framework_version).startswith(TENSORFLOW2_VERSION):
            model_name= "simple"
            model_path = os.path.join(serving_folder_path, "models", model_name)
            src_port = "8501"
            dst_port = "8501"
        else:
            src_port = grpc_port
            dst_port = "8500"

        docker_run_cmd = (
            f"{docker_cmd} run -id --name {container_name} -p {src_port}:{dst_port} "
            f"--device=/dev/neuron0 --net=host  --cap-add IPC_LOCK "
            f"--mount type=bind,source={model_path},target=/models/{model_name} -e TEST_MODE=1 -e MODEL_NAME={model_name} "
            f"-e NEURON_MONITOR_CW_REGION=us-east-1 -e NEURON_MONITOR_CW_NAMESPACE=tf1 "
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
            serving_folder_path, framework_version, ec2_connection, is_neuron, is_graviton, model_name, python_invoker
        )
        sleep(2)
        if not is_neuron:
            train_mnist_model(serving_folder_path, ec2_connection, python_invoker)
            sleep(10)
        ec2_connection.run(
            f"$(aws ecr get-login --no-include-email --region {region})", hide=True
        )
        LOGGER.info(docker_run_cmd)
        ec2_connection.run(docker_run_cmd, hide=True)
        sleep(20)
        if is_neuron and str(framework_version).startswith(TENSORFLOW2_VERSION):
            test_utils.request_tensorflow_inference(model_name, connection=ec2_connection, inference_string="'{\"instances\": [[1.0, 2.0, 5.0]]}'")
        else:
            test_utils.request_tensorflow_inference_grpc(
                script_file_path=mnist_client_path, port=grpc_port, connection=ec2_connection, ec2_instance_ami=ec2_instance_ami
            )
        if telemetry_mode:
            check_telemetry(ec2_connection, container_name)
    finally:
        ec2_connection.run(f"docker rm -f {container_name}", warn=True, hide=True)


def get_tensorflow_framework_version(image_uri):
    return re.findall(r"[1-2]\.[0-9][\d|\.]+", image_uri)[0]


def train_mnist_model(serving_folder_path, ec2_connection, python_invoker):
    ec2_connection.run(f"cd {serving_folder_path}")
    mnist_script_path = f"{serving_folder_path}/tensorflow_serving/example/mnist_saved_model.py"
    ec2_connection.run(
        f"{python_invoker} {mnist_script_path} {serving_folder_path}/models/mnist", hide=True
    )


def host_setup_for_tensorflow_inference(serving_folder_path, framework_version, ec2_connection, is_neuron, is_graviton, model_name, python_invoker):
    # Tensorflow 1.x doesn't have package with version 1.15.2 so use only 1.15
    if is_graviton:
        # TF training binary is used that is compatible for graviton instance type
        TF_URL="https://aws-dlc-graviton-training-binaries.s3.us-west-2.amazonaws.com/tensorflow/2.6.0/tensorflow-2.6.0-cp38-cp38-linux_aarch64.whl"
        ec2_connection.run(
            (
                f"{python_invoker} -m pip install --no-cache-dir -U {TF_URL}"
            ), hide=True
        )
        ec2_connection.run(
            (
                f"{python_invoker} -m pip install --no-dependencies --no-cache-dir tensorflow-serving-api=={framework_version}"
            ), hide=True
        )
    else: 
        ec2_connection.run(
            (
                f"{python_invoker} -m pip install --user -qq -U 'tensorflow<={framework_version}' "
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
            model_download = (
                f"wget -O {neuron_model_file} {model_file_path} "
            )
            ec2_connection.run(model_download)
    else:
        local_scripts_path = os.path.join("container_tests", "bin", "tensorflow_serving")
        ec2_connection.run(f"mkdir -p {serving_folder_path}")
        ec2_connection.run(f"cp -r {local_scripts_path} {serving_folder_path}")
        if is_neuron:
            neuron_local_model = os.path.join("$HOME", "container_tests", "bin", "neuron_tests", "simple")
            neuron_model_dir = os.path.join(serving_folder_path, "models")
            neuron_model_file_path = os.path.join(serving_folder_path, "models", "model_name", "1")
            LOGGER.info(f"Host Model path {neuron_model_file_path}")
            LOGGER.info(f"Host Model Dir {neuron_model_dir}")
            ec2_connection.run(f"mkdir -p {neuron_model_file_path}")
            ec2_connection.run(f"cp -r {neuron_local_model} {neuron_model_dir}")


def check_telemetry(ec2_connection, container_name):
    ec2_connection.run(f"docker exec -i {container_name} bash -c '[ -f /tmp/test_request.txt ]'")
    ec2_connection.run(f"docker exec -i {container_name} bash -c '[ -f /tmp/test_tag_request.txt ]'")
