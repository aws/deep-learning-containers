import logging
import pytest
import sys

from invoke import run

from test import test_utils

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stderr))


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
        LOGGER.info(docker_cmd)
        run_out = conn.run(docker_cmd, hide=True)
        LOGGER.info(f"{pytorch_inference} : {run_out.stdout}")
        if run_out.return_code != 0:
            LOGGER.info("docker run failed", run_out.return_code)
        output = conn.run("docker ps -a")
        LOGGER.info(f"containers for {pytorch_inference} : {output.stdout}")
        # inference_result = test_utils.request_pytorch_inference_densenet(conn)
        # assert inference_result, f"Failed to perform pytorch inference test for image: {pytorch_inference} on ec2"

    finally:
        output = run(f"docker rm -f {container_name}", warn=True, hide=True)
        LOGGER.info(f"{pytorch_inference} : {output.stdout}")
