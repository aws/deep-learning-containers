import pytest

from test import test_utils
from test.dlc_tests.conftest import LOGGER


SQUEEZENET_MODEL = "squeezenet"
BERT_MODEL = "bert_sst"

# @pytest.mark.parametrize("ec2_instance_type", ["p3.2xlarge"], indirect=True)
# def test_ec2_mxnet_inference_gpu(mxnet_inference, ec2_connection, region, gpu_only):
#     login_to_ecr_repo(ec2_connection, region)
#     run_ec2_mxnet_inference(mxnet_inference, SQUEEZENET_MODEL, "squeezenet", ec2_connection, "gpu", 80, 8081)
#     if "py3" in mxnet_inference:
#         run_ec2_mxnet_inference(mxnet_inference, BERT_MODEL, "gluonnlp", ec2_connection, "gpu", 90, 9091)

@pytest.mark.parametrize("ec2_instance_type", ["c5.4xlarge"], indirect=True)
def test_ec2_mxnet_inference_cpu(mxnet_inference, ec2_connection, region, cpu_only):
    login_to_ecr_repo(ec2_connection, region)
    if "py3" in mxnet_inference:
        run_ec2_mxnet_inference(mxnet_inference, SQUEEZENET_MODEL, "squeezenet", ec2_connection, "cpu", 80, 8081)
    # if "py3" in mxnet_inference:
    #     run_ec2_mxnet_inference(mxnet_inference, BERT_MODEL, "gluonnlp", ec2_connection, "cpu", 90, 9091)

def login_to_ecr_repo(ec2_connection, region):
    ec2_connection.run(
        f"$(aws ecr get-login --no-include-email --region {region})", hide=True
    )

def run_ec2_mxnet_inference(image_uri, model_name, container_tag, ec2_connection, processor, target_port, target_management_port):
    repo_name, image_tag = image_uri.split("/")[-1].split(":")
    container_name = f"{repo_name}-{image_tag}-ec2-{container_tag}"
    docker_cmd = "nvidia-docker" if "gpu" in image_uri else "docker"
    mms_inference_cmd = test_utils.get_mms_run_command(model_name, processor)
    docker_run_cmd = (
        f"{docker_cmd} run -itd --name {container_name}"
        f" -p {target_port}:8080 -p {target_management_port}:8081"
        f" {image_uri} {mms_inference_cmd}"
    )
    try:
        LOGGER.info(docker_run_cmd)
        ec2_connection.run(docker_run_cmd, hide=True)
        if model_name == SQUEEZENET_MODEL:
            inference_result = test_utils.request_mxnet_inference_squeezenet(
                connection=ec2_connection
            )
        elif model_name == BERT_MODEL:
            inference_result = test_utils.request_mxnet_inference_gluonnlp(
                connection=ec2_connection
            )
        assert (
            inference_result
        ), f"Failed to perform mxnet {model_name} inference test for image: {image_uri} on ec2"

    finally:
        ec2_connection.run(f"docker rm -f {container_name}", warn=True, hide=True)
