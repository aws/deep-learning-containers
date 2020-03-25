from time import sleep

from invoke import run
import pytest

from test.test_utils import get_image_framework_version
from test.test_utils.test_setup_utils import (host_setup_for_tensorflow_inference,
                                              request_tensorflow_inference_grpc,
                                              tensorflow_inference_test_cleanup)


def test_ec2_tensorflow_inference_grpc_cpu(tensorflow_inference, cpu_only):
    run(f"docker pull {tensorflow_inference}", hide="out")
    repo_name, image_tag = tensorflow_inference.split("/")[-1].split(":")
    container_name = f"{repo_name}-{image_tag}-container"
    framework_version = get_image_framework_version(tensorflow_inference)
    grpc_port = "8500"

    src_path, venv_path, model_location = host_setup_for_tensorflow_inference(container_name, framework_version)
    try:
        run(f"docker run -id --name {container_name} -p {grpc_port}:8500 "
            f"--mount type=bind,source={model_location},target=/models/mnist -e MODEL_NAME=mnist "
            f"{tensorflow_inference}", echo=True)

        sleep(30)

        request_tensorflow_inference_grpc(src_path, venv_path, port=grpc_port)
    finally:
        run(f"docker rm -f {container_name}", warn=True, echo=True)
        tensorflow_inference_test_cleanup(src_path, venv_path)


@pytest.mark.skip("nvidia-docker issues in CodeBuild")
def test_ec2_tensorflow_inference_grpc_gpu(tensorflow_inference, gpu_only):
    run(f"docker pull {tensorflow_inference}", hide="out")
    repo_name, image_tag = tensorflow_inference.split("/")[-1].split(":")
    container_name = f"{repo_name}-{image_tag}-container"
    framework_version = get_image_framework_version(tensorflow_inference)
    grpc_port = "8500"

    src_path, venv_path, model_location = host_setup_for_tensorflow_inference(container_name, framework_version)
    try:
        run(f"nvidia-docker run -id --name {container_name} -p {grpc_port}:8500 "
            f"--mount type=bind,source={model_location},target=/models/mnist -e MODEL_NAME=mnist "
            f"{tensorflow_inference}")

        sleep(30)

        request_tensorflow_inference_grpc(src_path, venv_path, port=grpc_port)
    finally:
        run(f"docker rm -f {container_name}", warn=True)
        tensorflow_inference_test_cleanup(src_path, venv_path)
