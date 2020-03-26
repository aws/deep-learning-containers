from time import sleep

from invoke import run

from test.test_utils import request_tensorflow_inference_grpc
import test.test_utils.ec2 as ec2_utils


def test_ec2_tensorflow_inference_grpc(tensorflow_inference, mnist_serving_model, cpu_only):
    run(f"docker pull {tensorflow_inference}", hide="out")
    src_path, model_location = mnist_serving_model

    repo_name, image_tag = tensorflow_inference.split("/")[-1].split(":")
    container_name = f"{repo_name}-{image_tag}-container"
    grpc_port = ec2_utils.get_inference_ec2_test_port_number(tensorflow_inference)
    docker_bin = "nvidia-docker" if "gpu" in tensorflow_inference else "docker"

    try:
        run(f"{docker_bin} run -id --name {container_name} -p {grpc_port}:8500 "
            f"--mount type=bind,source={model_location},target=/models/mnist -e MODEL_NAME=mnist "
            f"{tensorflow_inference}", echo=True)

        sleep(30)

        request_tensorflow_inference_grpc(src_path, port=grpc_port)
    finally:
        run(f"docker rm -f {container_name}", warn=True, echo=True)
