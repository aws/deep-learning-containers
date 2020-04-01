import pytest

from test import test_utils
from test.dlc_tests.conftest import LOGGER


@pytest.mark.parametrize("ec2_instance_type", ["p3.2xlarge"], indirect=True)
def test_ec2_pytorch_inference_gpu(pytorch_inference, ec2_connection, region, gpu_only):
    repo_name, image_tag = pytorch_inference.split("/")[-1].split(":")
    container_name = f"{repo_name}-{image_tag}-ec2"
    model_name = "pytorch-densenet"
    mms_inference_cmd =test_utils.get_mms_run_command(model_name, "gpu")

    docker_cmd = (
        f"docker run -itd --name {container_name}"
        f" -p 80:8080 -p 8081:8081"
        f" {pytorch_inference} {mms_inference_cmd}"
    )
    try:
        ec2_connection.run(f"$(aws ecr get-login --no-include-email --region {region})", hide=True)
        LOGGER.info(docker_cmd)
        ec2_connection.run(docker_cmd, hide=True)
        inference_result = test_utils.request_pytorch_inference_densenet(connection=ec2_connection)
        assert inference_result, f"Failed to perform pytorch inference test for image: {pytorch_inference} on ec2"

    finally:
        ec2_connection.run(f"docker rm -f {container_name}", warn=True, hide=True)


@pytest.mark.parametrize("ec2_instance_type", ["c5.4xlarge"], indirect=True)
def test_ec2_pytorch_inference_cpu(pytorch_inference, ec2_connection, region, cpu_only):
    repo_name, image_tag = pytorch_inference.split("/")[-1].split(":")
    container_name = f"{repo_name}-{image_tag}-ec2"
    model_name = "pytorch-densenet"
    mms_inference_cmd = test_utils.get_mms_run_command(model_name, "cpu")
    docker_cmd = (
        f"docker run -itd --name {container_name}"
        f" -p 80:8080 -p 8081:8081"
        f" {pytorch_inference} {mms_inference_cmd}"
    )
    try:
        ec2_connection.run(f"$(aws ecr get-login --no-include-email --region {region})", hide=True)
        LOGGER.info(docker_cmd)
        ec2_connection.run(docker_cmd, hide=True)
        inference_result = test_utils.request_pytorch_inference_densenet(connection=ec2_connection)
        assert inference_result, f"Failed to perform pytorch inference test for image: {pytorch_inference} on ec2"

    finally:
        ec2_connection.run(f"docker rm -f {container_name}", warn=True, hide=True)
