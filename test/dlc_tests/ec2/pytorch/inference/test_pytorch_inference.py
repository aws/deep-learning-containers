import logging
import pytest
import sys

from invoke import run

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stderr))


def get_mms_inference_cmd():
    model_name_with_location = "pytorch-densenet=https://asimov-multi-model-server.s3.amazonaws.com/pytorch/densenet/densenet.mar"
    mms_inference_cmd = "mxnet-model-server --start --mms-config /home/model-server/config.properties --models"
    return f"{mms_inference_cmd} {model_name_with_location}"


@pytest.mark.parametrize("ec2_instance_type", ["p3.2xlarge"], indirect=True)
def test_ec2_pytorch_inference_gpu(pytorch_inference, ec2_connection, gpu_only):
    conn = ec2_connection()

    # Assert that connection is successful
    output = conn.run(f"echo {pytorch_inference}").stdout.strip("\n")
    assert output == pytorch_inference, f"Fabric output did not match -- {output}"

    repo_name, image_tag = pytorch_inference.split("/")[-1].split(":")
    container_name = f"{repo_name}-{image_tag}-ec2"
    mms_inference_cmd = get_mms_inference_cmd()

    docker_cmd = f"docker run -itd --name {container_name} --mount type=bind,src=$(pwd)/container_tests,target=/test --entrypoint='/bin/bash -p 80:8080  -p 8081:8081 {mms_inference_cmd}"
    try:
        run_out = run(docker_cmd, hide=True)
        LOGGER.info(run_out.stdout)
        assert (
            run_out.return_code == 0
        ), f"Failed to perform pytorch inference test for {pytorch_inference} on ec2"

    finally:
        run(f"docker rm -f {container_name}", hide=True)


@pytest.mark.parametrize("ec2_instance_type", ["c5.4xlarge"], indirect=True)
def test_ec2_pytorch_inference_cpu(pytorch_inference, ec2_connection, cpu_only):
    conn = ec2_connection()

    # Assert that connection is successful
    output = conn.run(f"echo {pytorch_inference}").stdout.strip("\n")
    assert output == pytorch_inference, f"Fabric output did not match -- {output}"

    repo_name, image_tag = pytorch_inference.split("/")[-1].split(":")
    container_name = f"{repo_name}-{image_tag}-ec2"
    mms_inference_cmd = get_mms_inference_cmd()
    docker_cmd = (
        f"docker run -itd --name {container_name} "
        f"--mount type=bind,src=$(pwd)/container_tests,target=/test "
        f"--entrypoint='/bin/bash -p 80:8080  "
        f"-p 8081:8081 {mms_inference_cmd}"
    )
    try:
        run_out = run(docker_cmd, hide=True)
        LOGGER.info(run_out.stdout)
        assert (
            run_out.return_code == 0
        ), f"Failed to perform pytorch inference test for {pytorch_inference} on ec2"

    finally:
        run(f"docker rm -f {container_name}", hide=True)
