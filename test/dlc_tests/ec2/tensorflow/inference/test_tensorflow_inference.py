import os
import re
from time import sleep
import pytest

from packaging.version import Version
from packaging.specifiers import SpecifierSet

import test.test_utils.ec2 as ec2_utils

from test import test_utils
from test.test_utils.ec2 import (
    get_ec2_instance_type,
    get_ec2_accelerator_type,
    execute_ec2_telemetry_test,
)
from test.dlc_tests.conftest import LOGGER
from test.test_utils import CONTAINER_TESTS_PREFIX

TENSORFLOW1_VERSION = "1."
TENSORFLOW2_VERSION = "2."


TF_EC2_GPU_INSTANCE_TYPE = get_ec2_instance_type(default="g4dn.8xlarge", processor="gpu")
TF_EC2_CPU_INSTANCE_TYPE = get_ec2_instance_type(default="c5.4xlarge", processor="cpu")
TF_EC2_EIA_ACCELERATOR_TYPE = get_ec2_accelerator_type(default="eia1.large", processor="eia")
TF_EC2_NEURON_ACCELERATOR_TYPE = get_ec2_instance_type(default="inf1.xlarge", processor="neuron")
TF_EC2_NEURONX_ACCELERATOR_TYPE = get_ec2_instance_type(default="trn1.2xlarge", processor="neuronx")
TF_TELEMETRY_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testTelemetry")
TF_EC2_NEURONX_INF2_ACCELERATOR_TYPE = get_ec2_instance_type(
    default="inf2.xlarge", processor="neuronx"
)
TF_EC2_SINGLE_GPU_INSTANCE_TYPE = get_ec2_instance_type(
    default="g5.8xlarge",
    processor="gpu",
    filter_function=ec2_utils.filter_only_single_gpu,
)
TF_EC2_GRAVITON_INSTANCE_TYPE = get_ec2_instance_type(
    default="c6g.4xlarge", processor="cpu", arch_type="graviton"
)
TF_EC2_ARM64_INSTANCE_TYPE = get_ec2_instance_type(
    default="c6g.4xlarge", processor="cpu", arch_type="arm64"
)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.skipif(
    not test_utils.is_deep_canary_context() or not os.getenv("REGION") == "us-west-2",
    reason="This test only needs to run in deep-canary context in us-west-2",
)
@pytest.mark.deep_canary("Reason: This test is a simple tf mnist test")
@pytest.mark.model("mnist")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_ec2_tensorflow_inference_gpu_deep_canary(
    tensorflow_inference, ec2_connection, region, gpu_only
):
    if ":2.14" in tensorflow_inference:
        # TF 2.14 deep canaries are failing due to numpy mismatch
        ec2_connection.run("pip install numpy==1.26.4")
    run_ec2_tensorflow_inference(tensorflow_inference, ec2_connection, "8500", region)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.skipif(
    not test_utils.is_deep_canary_context() or not os.getenv("REGION") == "us-west-2",
    reason="This test only needs to run in deep-canary context in us-west-2",
)
@pytest.mark.deep_canary("Reason: This test is a simple tf mnist test")
@pytest.mark.model("mnist")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_ec2_tensorflow_inference_cpu_deep_canary(
    tensorflow_inference, ec2_connection, region, cpu_only
):
    if ":2.14" in tensorflow_inference:
        # TF 2.14 deep canaries are failing due to numpy mismatch
        ec2_connection.run("pip install numpy==1.26.4")
    run_ec2_tensorflow_inference(tensorflow_inference, ec2_connection, "8500", region)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.skipif(
    not test_utils.is_deep_canary_context() or not os.getenv("REGION") == "us-west-2",
    reason="This test only needs to run in deep-canary context in us-west-2",
)
@pytest.mark.deep_canary("Reason: This test is a simple tf mnist test")
@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_GRAVITON_INSTANCE_TYPE, indirect=True)
@pytest.mark.parametrize(
    "ec2_instance_ami", [test_utils.AL2023_BASE_DLAMI_ARM64_US_WEST_2], indirect=True
)
def test_ec2_tensorflow_inference_graviton_cpu_deep_canary(
    tensorflow_inference_graviton, ec2_connection, region, cpu_only
):
    run_ec2_tensorflow_inference(tensorflow_inference_graviton, ec2_connection, "8500", region)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.skipif(
    not test_utils.is_deep_canary_context() or not os.getenv("REGION") == "us-west-2",
    reason="This test only needs to run in deep-canary context in us-west-2",
)
@pytest.mark.deep_canary("Reason: This test is a simple tf mnist test")
@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_ARM64_INSTANCE_TYPE, indirect=True)
@pytest.mark.parametrize(
    "ec2_instance_ami", [test_utils.AL2023_BASE_DLAMI_ARM64_US_WEST_2], indirect=True
)
def test_ec2_tensorflow_inference_arm64_cpu_deep_canary(
    tensorflow_inference_arm64, ec2_connection, region, cpu_only
):
    run_ec2_tensorflow_inference(tensorflow_inference_arm64, ec2_connection, "8500", region)


@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_NEURON_ACCELERATOR_TYPE, indirect=True)
@pytest.mark.parametrize("ec2_instance_ami", [test_utils.UL22_BASE_NEURON_US_WEST_2], indirect=True)
@pytest.mark.team("neuron")
def test_ec2_tensorflow_inference_neuron(tensorflow_inference_neuron, ec2_connection, region):
    run_ec2_tensorflow_inference(tensorflow_inference_neuron, ec2_connection, "8500", region)


@pytest.mark.model("mnist")
@pytest.mark.parametrize(
    "ec2_instance_type",
    TF_EC2_NEURONX_ACCELERATOR_TYPE + TF_EC2_NEURONX_INF2_ACCELERATOR_TYPE,
    indirect=True,
)
@pytest.mark.team("neuron")
@pytest.mark.parametrize("ec2_instance_ami", [test_utils.UL22_BASE_NEURON_US_WEST_2], indirect=True)
def test_ec2_tensorflow_inference_neuronx(tensorflow_inference_neuronx, ec2_connection, region):
    run_ec2_tensorflow_inference(tensorflow_inference_neuronx, ec2_connection, "8500", region)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("mnist")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_ec2_tensorflow_inference_gpu(
    tensorflow_inference, ec2_connection, region, gpu_only, ec2_instance_type
):
    if test_utils.is_image_incompatible_with_instance_type(tensorflow_inference, ec2_instance_type):
        pytest.skip(
            f"Image {tensorflow_inference} is incompatible with instance type {ec2_instance_type}"
        )
    run_ec2_tensorflow_inference(tensorflow_inference, ec2_connection, "8500", region)


# Start from Tf2.18, cuda build doesn't support tensorRT anymore
@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("N/A")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_ec2_tensorflow_inference_gpu_tensorrt(
    tensorflow_inference, ec2_connection, region, gpu_only, ec2_instance_type, below_tf218_only
):
    if test_utils.is_image_incompatible_with_instance_type(tensorflow_inference, ec2_instance_type):
        pytest.skip(
            f"Image {tensorflow_inference} is incompatible with instance type {ec2_instance_type}"
        )
    _, framework_version = test_utils.get_framework_and_version_from_tag(tensorflow_inference)
    home_dir = ec2_connection.run("echo $HOME").stdout.strip("\n")
    serving_folder_path = os.path.join(home_dir, "serving")
    build_container_name = "tensorrt-build-container"
    serving_container_name = "tensorrt-serving-container"
    model_name = "tftrt_saved_model"
    model_creation_script_folder = os.path.join(
        serving_folder_path, "tensorflow_serving", "example"
    )
    model_path = os.path.join(
        serving_folder_path, "tensorflow_serving", "example", "models", model_name
    )

    # Use helper function pull_tensorrt_build_image to get the closest matching major.minor.patch
    # version for a particular TF inference framework version. Sometimes TF serving versions
    # are a patch version or two ahead of the corresponding TF version.
    upstream_build_image_uri = pull_tensorrt_build_image(ec2_connection, framework_version)
    docker_build_model_command = (
        f"docker run --runtime=nvidia --gpus all --rm --name {build_container_name} "
        f"-v {model_creation_script_folder}:/script_folder/ -i {upstream_build_image_uri} "
        f"python /script_folder/create_tensorrt_model.py"
    )
    docker_run_server_cmd = (
        f"docker run --runtime=nvidia --gpus all -id --name {serving_container_name} -p 8501:8501 "
        f"--mount type=bind,source={model_path},target=/models/{model_name}/1 -e TEST_MODE=1 -e MODEL_NAME={model_name}"
        f" {tensorflow_inference}"
    )

    try:
        account_id = test_utils.get_account_id_from_image_uri(tensorflow_inference)
        test_utils.login_to_ecr_registry(ec2_connection, account_id, region)
        host_setup_for_tensorflow_inference(serving_folder_path, framework_version, ec2_connection)
        sleep(2)

        ## Build TensorRt Model
        ec2_connection.run(docker_build_model_command, hide=True)

        ## Run Model Server
        ec2_connection.run(docker_run_server_cmd, hide=True)
        test_results = test_utils.request_tensorflow_inference(
            model_name,
            connection=ec2_connection,
            inference_string=f"""'{{"instances": [[{",".join([str([1]*28)]*28)}]]}}'""",
        )
        assert test_results, "TensorRt test failed!"
    except Exception:
        remote_out = ec2_connection.run(
            f"docker logs {serving_container_name}", warn=True, hide=True
        )
        LOGGER.info(
            f"--- TF container logs ---\n--- STDOUT ---\n{remote_out.stdout}\n--- STDERR ---\n{remote_out.stderr}"
        )
        raise
    finally:
        ec2_connection.run(f"docker rm -f {serving_container_name}", warn=True, hide=True)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("mnist")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_ec2_tensorflow_inference_cpu(tensorflow_inference, ec2_connection, region, cpu_only):
    run_ec2_tensorflow_inference(tensorflow_inference, ec2_connection, "8500", region)


@pytest.mark.integration("elastic_inference")
@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
@pytest.mark.parametrize("ei_accelerator_type", TF_EC2_EIA_ACCELERATOR_TYPE, indirect=True)
def test_ec2_tensorflow_inference_eia_cpu(tensorflow_inference_eia, ec2_connection, region):
    run_ec2_tensorflow_inference(tensorflow_inference_eia, ec2_connection, "8500", region)


@pytest.mark.integration("elastic_inference")
@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
@pytest.mark.parametrize("ei_accelerator_type", TF_EC2_EIA_ACCELERATOR_TYPE, indirect=True)
def test_ec2_tensorflow_inference_eia_gpu(
    tensorflow_inference_eia, ec2_connection, region, ec2_instance_type
):
    if ec2_instance_type == "p4d.24xlarge":
        pytest.skip(
            f"Skipping EIA GPU test for {ec2_instance_type} instance type. See https://github.com/aws/deep-learning-containers/issues/962"
        )
    run_ec2_tensorflow_inference(tensorflow_inference_eia, ec2_connection, "8500", region)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("mnist")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_SINGLE_GPU_INSTANCE_TYPE, indirect=True)
def test_ec2_tensorflow_inference_telemetry_framework_gpu(
    tensorflow_inference, ec2_connection, region, gpu_only, ec2_instance_type
):
    if test_utils.is_image_incompatible_with_instance_type(tensorflow_inference, ec2_instance_type):
        pytest.skip(
            f"Image {tensorflow_inference} is incompatible with instance type {ec2_instance_type}"
        )
    run_ec2_tensorflow_inference(tensorflow_inference, ec2_connection, "8500", region, True)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("mnist")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_SINGLE_GPU_INSTANCE_TYPE, indirect=True)
def test_ec2_tensorflow_inference_telemetry_bashrc_gpu(
    tensorflow_inference, ec2_connection, region, gpu_only, ec2_instance_type
):
    if test_utils.is_image_incompatible_with_instance_type(tensorflow_inference, ec2_instance_type):
        pytest.skip(
            f"Image {tensorflow_inference} is incompatible with instance type {ec2_instance_type}"
        )
    execute_ec2_telemetry_test(
        ec2_connection,
        tensorflow_inference,
        "bashrc",
        "tensorflow_inf_telemetry",
        TF_TELEMETRY_CMD,
        opt_in=True,
    )
    execute_ec2_telemetry_test(
        ec2_connection,
        tensorflow_inference,
        "bashrc",
        "tensorflow_inf_telemetry",
        TF_TELEMETRY_CMD,
        opt_in=False,
    )


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("mnist")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_ec2_tensorflow_inference_telemetry_framework_cpu(
    tensorflow_inference, ec2_connection, region, cpu_only
):
    run_ec2_tensorflow_inference(tensorflow_inference, ec2_connection, "8500", region, True)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("mnist")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_ec2_tensorflow_inference_telemetry_bashrc_cpu(
    tensorflow_inference, ec2_connection, region, cpu_only
):
    execute_ec2_telemetry_test(
        ec2_connection,
        tensorflow_inference,
        "bashrc",
        "tensorflow_inf_telemetry",
        TF_TELEMETRY_CMD,
        opt_in=True,
    )
    execute_ec2_telemetry_test(
        ec2_connection,
        tensorflow_inference,
        "bashrc",
        "tensorflow_inf_telemetry",
        TF_TELEMETRY_CMD,
        opt_in=False,
    )


@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_GRAVITON_INSTANCE_TYPE, indirect=True)
@pytest.mark.parametrize(
    "ec2_instance_ami", [test_utils.AL2023_BASE_DLAMI_ARM64_US_WEST_2], indirect=True
)
def test_ec2_tensorflow_inference_graviton_cpu(
    tensorflow_inference_graviton, ec2_connection, region, cpu_only
):
    run_ec2_tensorflow_inference(tensorflow_inference_graviton, ec2_connection, "8500", region)


@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_ARM64_INSTANCE_TYPE, indirect=True)
@pytest.mark.parametrize(
    "ec2_instance_ami", [test_utils.AL2023_BASE_DLAMI_ARM64_US_WEST_2], indirect=True
)
def test_ec2_tensorflow_inference_arm64_cpu(
    tensorflow_inference_arm64, ec2_connection, region, cpu_only
):
    run_ec2_tensorflow_inference(tensorflow_inference_arm64, ec2_connection, "8500", region)


@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_GRAVITON_INSTANCE_TYPE, indirect=True)
@pytest.mark.parametrize(
    "ec2_instance_ami", [test_utils.AL2023_BASE_DLAMI_ARM64_US_WEST_2], indirect=True
)
def test_ec2_tensorflow_inference_graviton_cpu_telemetry(
    tensorflow_inference_graviton, ec2_connection, region, cpu_only
):
    run_ec2_tensorflow_inference(
        tensorflow_inference_graviton, ec2_connection, "8500", region, True
    )


@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_ARM64_INSTANCE_TYPE, indirect=True)
@pytest.mark.parametrize(
    "ec2_instance_ami", [test_utils.AL2023_BASE_DLAMI_ARM64_US_WEST_2], indirect=True
)
def test_ec2_tensorflow_inference_arm64_telemetry_framework_cpu(
    tensorflow_inference_arm64, ec2_connection, region, cpu_only
):
    run_ec2_tensorflow_inference(tensorflow_inference_arm64, ec2_connection, "8500", region, True)


def run_ec2_tensorflow_inference(
    image_uri, ec2_connection, grpc_port, region, telemetry_mode=False
):
    repo_name, image_tag = image_uri.split("/")[-1].split(":")
    container_name = f"{repo_name}-{image_tag}-ec2"
    _, framework_version = test_utils.get_framework_and_version_from_tag(image_uri)
    home_dir = ec2_connection.run("echo $HOME").stdout.strip("\n")
    serving_folder_path = os.path.join(home_dir, "serving")
    model_name = "mnist"
    model_path = os.path.join(serving_folder_path, "models", model_name)
    mnist_client_path = os.path.join(
        serving_folder_path, "tensorflow_serving", "example", "mnist_client.py"
    )

    is_neuron = "neuron" in image_uri
    is_neuron_x = "neuronx" in image_uri
    is_arm64 = "graviton" in image_uri or "arm64" in image_uri

    docker_runtime = "--runtime=nvidia --gpus all" if "gpu" in image_uri else ""

    if is_neuron:
        # For 2.5 using rest api port instead of grpc since using curl for prediction instead of grpc
        if str(framework_version).startswith(TENSORFLOW2_VERSION):
            model_name = "simple_x" if is_neuron_x else "simple"
            model_path = os.path.join(serving_folder_path, "models", model_name)
            src_port = "8501"
            dst_port = "8501"
        else:
            src_port = grpc_port
            dst_port = "8500"

        docker_run_cmd = (
            f"docker run {docker_runtime} -id --name {container_name} -p {src_port}:{dst_port} "
            f"--device=/dev/neuron0 --net=host  --cap-add IPC_LOCK "
            f"--mount type=bind,source={model_path},target=/models/{model_name} -e TEST_MODE=1 -e MODEL_NAME={model_name} "
            f"-e NEURON_MONITOR_CW_REGION=us-east-1 -e NEURON_MONITOR_CW_NAMESPACE=tf1 "
            f" {image_uri}"
        )
    else:
        docker_run_cmd = (
            f"docker run {docker_runtime} -id --name {container_name} -p {grpc_port}:8500 "
            f"--mount type=bind,source={model_path},target=/models/mnist -e TEST_MODE=1 -e MODEL_NAME=mnist"
            f" {image_uri}"
        )

    try:
        host_setup_for_tensorflow_inference(
            serving_folder_path,
            framework_version,
            ec2_connection,
            is_neuron,
            is_arm64,
            model_name,
        )
        sleep(2)
        if not is_neuron:
            train_mnist_model(serving_folder_path, ec2_connection)
            sleep(10)
        account_id = test_utils.get_account_id_from_image_uri(image_uri)
        test_utils.login_to_ecr_registry(ec2_connection, account_id, region)
        ec2_connection.run(docker_run_cmd, hide=True)
        sleep(20)
        if is_neuron and str(framework_version).startswith(TENSORFLOW2_VERSION):
            test_utils.request_tensorflow_inference(
                model_name,
                connection=ec2_connection,
                inference_string="'{\"instances\": [[1.0, 2.0, 5.0]]}'",
            )
        else:
            test_utils.request_tensorflow_inference_grpc(
                script_file_path=mnist_client_path, port=grpc_port, connection=ec2_connection
            )
        if telemetry_mode:
            check_telemetry(ec2_connection, container_name)
    except Exception:
        remote_out = ec2_connection.run(f"docker logs {container_name}", warn=True, hide=True)
        LOGGER.info(
            f"--- TF container logs ---\n--- STDOUT ---\n{remote_out.stdout}\n--- STDERR ---\n{remote_out.stderr}"
        )
        raise
    finally:
        ec2_connection.run(f"docker rm -f {container_name}", warn=True, hide=True)


def train_mnist_model(serving_folder_path, ec2_connection):
    ec2_connection.run(f"cd {serving_folder_path}")
    mnist_script_path = f"{serving_folder_path}/tensorflow_serving/example/mnist_saved_model.py"
    ec2_connection.run(f"python {mnist_script_path} {serving_folder_path}/models/mnist", hide=True)


def host_setup_for_tensorflow_inference(
    serving_folder_path,
    framework_version,
    ec2_connection,
    is_neuron=False,
    is_arm64=False,
    model_name=None,
):
    # Attempting a pin will result in pip not finding the version. The internal repo only has a custom Tensorflow 2.6
    # which is not compatible with TF 2.9+ and this is the recommended action.
    if is_arm64:
        ec2_connection.run(f"pip install --no-cache-dir -U tensorflow-cpu-aws", hide=True)

        # If framework_version is only major.minor, then append .* for tensorflow-serving-api installation
        tfs_api_version = framework_version
        if re.fullmatch(r"\d+\.\d+", tfs_api_version):
            tfs_api_version += ".*"

        # Removed the protobuf version constraint because it prevents the matching version
        # of tensorflow and tensorflow-serving-api from being installed.
        # If we face protobuf-related version mismatch issues in the future,
        # please add a constraint at the necessary version back to this code, such as
        # 'protobuf>=3.20,<3.21'.
        ec2_connection.run(
            (
                f"pip install --no-dependencies --no-cache-dir "
                f"'tensorflow-serving-api=={tfs_api_version}'"
            ),
            hide=True,
        )
    else:
        # Removed the protobuf version constraint because it prevents the matching version
        # of tensorflow and tensorflow-serving-api from being installed.
        # If we face protobuf-related version mismatch issues in the future,
        # please add a constraint at the necessary version back to this code, such as
        # 'protobuf>=3.20,<3.21'.
        ec2_connection.run(
            (
                f"pip install --user -qq -U 'tensorflow<={framework_version}' "
                f" 'tensorflow-serving-api<={framework_version}'"
            ),
            hide=True,
        )
    if os.path.exists(f"{serving_folder_path}"):
        ec2_connection.run(f"rm -rf {serving_folder_path}")
    if str(framework_version).startswith(TENSORFLOW1_VERSION):
        run_out = ec2_connection.run(
            f"git clone https://github.com/tensorflow/serving.git {serving_folder_path}"
        )
        git_branch_version = re.findall(r"[1-2]\.[0-9]\d", framework_version)[0]
        ec2_connection.run(f"cd {serving_folder_path} && git checkout r{git_branch_version}")
        LOGGER.info(f"Clone TF serving repository status {run_out.return_code == 0}")
        if is_neuron:
            container_test_local_file = os.path.join(
                "$HOME", "container_tests/bin/neuron_tests/mnist_client.py"
            )
            ec2_connection.run(
                f"cp -f {container_test_local_file} {serving_folder_path}/tensorflow_serving/example"
            )
            neuron_model_file_path = os.path.join(serving_folder_path, f"models/{model_name}/1")
            neuron_model_file = os.path.join(neuron_model_file_path, "saved_model.pb")
            LOGGER.info(f"Host Model path {neuron_model_file_path}")
            ec2_connection.run(f"mkdir -p {neuron_model_file_path}")
            model_file_path = f"https://aws-dlc-sample-models.s3.amazonaws.com/{model_name}_neuron/1/saved_model.pb"
            model_download = f"wget -O {neuron_model_file} {model_file_path} "
            ec2_connection.run(model_download)
    else:
        local_scripts_path = os.path.join("container_tests", "bin", "tensorflow_serving")
        ec2_connection.run(f"mkdir -p {serving_folder_path}")
        ec2_connection.run(f"cp -r {local_scripts_path} {serving_folder_path}")
        if is_neuron:
            neuron_local_model = os.path.join(
                "$HOME", "container_tests", "bin", "neuron_tests", model_name
            )
            neuron_model_dir = os.path.join(serving_folder_path, "models")
            neuron_model_file_path = os.path.join(serving_folder_path, "models", model_name, "1")
            LOGGER.info(f"Host Model path {neuron_model_file_path}")
            LOGGER.info(f"Host Model Dir {neuron_model_dir}")
            ec2_connection.run(f"mkdir -p {neuron_model_file_path}")
            ec2_connection.run(f"cp -r {neuron_local_model} {neuron_model_dir}")


def check_telemetry(ec2_connection, container_name):
    ec2_connection.run(f"docker exec -i {container_name} bash -c '[ -f /tmp/test_request.txt ]'")
    ec2_connection.run(
        f"docker exec -i {container_name} bash -c '[ -f /tmp/test_tag_request.txt ]'"
    )


def pull_tensorrt_build_image(ec2_connection, framework_version):
    """
    Download tensorflow/tensorflow:<framework_version>-gpu image used for tensorrt model builds.
    Since tensorflow/tensorflow does not always have all patch versions of a particular TF version,
    we want to download the closest available version below the patch version of the TF DLC image.

    :param ec2_connection: fabric.Connection object
    :param framework_version: str framework version of TF image being tested
    :return: str tensorrt build image tag
    """
    tf_image_version = Version(framework_version)
    if tf_image_version in SpecifierSet("==2.14.*"):
        # Add a special case for TF 2.14 to account for mismatch in TensorRT version between
        # tensorflow/tensorflow:2.14 (8.6.1) and tensorflow/serving:2.14 (8.4.3).
        # Do not change the range of versions covered by the SpecifierSet in the if-condition
        # unless it is confirmed that TF images for the newer framework version have the same issue.
        return "tensorflow/tensorflow:2.13.0-gpu"
    patch_version = tf_image_version.micro
    upstream_build_image_uri = f"tensorflow/tensorflow:{framework_version}-gpu"
    while patch_version >= 0:
        major_version = tf_image_version.major
        minor_version = tf_image_version.minor
        tf_image_version = Version(f"{major_version}.{minor_version}.{patch_version}")
        upstream_build_image_uri = f"tensorflow/tensorflow:{tf_image_version.base_version}-gpu"
        response = ec2_connection.run(
            f"docker pull {upstream_build_image_uri}", warn=True, hide=True
        )
        if response.failed:
            patch_version -= 1
        else:
            break
    if patch_version < 0:
        raise RuntimeError(
            f"Failed to pull an image for tensorflow/tensorflow matching {framework_version}"
        )
    return upstream_build_image_uri
