import logging
import pytest
import sys

from invoke import run

from test import test_utils

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stderr))


@pytest.mark.parametrize("ec2_instance_type", ["p3.2xlarge"], indirect=True)
def test_ec2_pytorch_inference_gpu(pytorch_inference, ec2_connection, gpu_only):
    conn = ec2_connection

    repo_name, image_tag = pytorch_inference.split("/")[-1].split(":")
    container_name = f"{repo_name}-{image_tag}-ec2"
    model_name = "pytorch-densenet"
    mms_inference_cmd =test_utils.get_mms_run_command(model_name, "gpu")

    docker_cmd = (
        f"docker run -itd --name {container_name}"
        f" --mount type=bind,src=$(pwd)/container_tests,target=/test"
        f" -p 80:8080 -p 8081:8081"
        f" {pytorch_inference} {mms_inference_cmd}"
    )
    try:
        conn.run(f"$(aws ecr get-login --no-include-email --region {test_utils.DEFAULT_REGION})", hide=True)
        conn.run(docker_cmd, hide=True)
        inference_result = test_utils.request_pytorch_inference_densenet(conn)
        assert inference_result, f"Failed to perform pytorch inference test for image: {pytorch_inference} on ec2"

    finally:
        conn.run(f"docker rm -f {container_name}", hide=True)


@pytest.mark.parametrize("ec2_instance_type", ["c5.4xlarge"], indirect=True)
def test_ec2_pytorch_inference_cpu(pytorch_inference, ec2_connection, cpu_only):
    conn = ec2_connection

    repo_name, image_tag = pytorch_inference.split("/")[-1].split(":")
    container_name = f"{repo_name}-{image_tag}-ec2"
    model_name = "pytorch-densenet"
    mms_inference_cmd = test_utils.get_mms_run_command(model_name, "cpu")
    docker_cmd = (
        f"docker run -itd --name {container_name}"
        f" --mount type=bind,src=$(pwd)/container_tests,target=/test"
        f" -p 80:8080 -p 8081:8081"
        f" {pytorch_inference} {mms_inference_cmd}"
    )
    try:
        conn.run(f"$(aws ecr get-login --no-include-email --region {test_utils.DEFAULT_REGION})", hide=True)
        conn.run(docker_cmd, hide=True)
        inference_result = test_utils.request_pytorch_inference_densenet(conn)
        assert inference_result, f"Failed to perform pytorch inference test for image: {pytorch_inference} on ec2"

    finally:
        run(f"docker rm -f {container_name}", hide=True)
