import os
import pytest

from test import test_utils
from test.test_utils import CONTAINER_TESTS_PREFIX
from test.test_utils.ec2 import get_ec2_instance_type, execute_ec2_inference_test, get_ec2_accelerator_type
from test.dlc_tests.conftest import LOGGER


SQUEEZENET_MODEL = "squeezenet"
BERT_MODEL = "bert_sst"
RESNET_EIA_MODEL = "resnet-152-eia"


MX_EC2_GPU_INSTANCE_TYPE = get_ec2_instance_type(default="g3.8xlarge", processor="gpu")
MX_EC2_CPU_INSTANCE_TYPE = get_ec2_instance_type(default="c5.4xlarge", processor="cpu")
MX_EC2_EIA_ACCELERATOR_TYPE = get_ec2_accelerator_type(default="eia1.large", processor="eia")
MX_TELEMETRY_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "test_mx_dlc_telemetry_test")


@pytest.mark.model(SQUEEZENET_MODEL)
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_ec2_mxnet_squeezenet_inference_gpu(mxnet_inference, ec2_connection, region, gpu_only):
    run_ec2_mxnet_inference(mxnet_inference, SQUEEZENET_MODEL, "squeezenet", ec2_connection, "gpu", region, 80, 8081)


@pytest.mark.integration("gluonnlp")
@pytest.mark.model(BERT_MODEL)
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_ec2_mxnet_gluonnlp_inference_gpu(mxnet_inference, ec2_connection, region, gpu_only, py3_only):
    run_ec2_mxnet_inference(mxnet_inference, BERT_MODEL, "gluonnlp", ec2_connection, "gpu", region, 90, 9091)


@pytest.mark.model(SQUEEZENET_MODEL)
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_ec2_mxnet_squeezenet_inference_cpu(mxnet_inference, ec2_connection, region, cpu_only):
    run_ec2_mxnet_inference(mxnet_inference, SQUEEZENET_MODEL, "squeezenet", ec2_connection, "cpu", region, 80, 8081)


@pytest.mark.integration("elastic_inference")
@pytest.mark.model(RESNET_EIA_MODEL)
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_CPU_INSTANCE_TYPE, indirect=True)
@pytest.mark.parametrize("ei_accelerator_type", MX_EC2_EIA_ACCELERATOR_TYPE, indirect=True)
def test_ec2_mxnet_resnet_inference_eia_cpu(mxnet_inference_eia, ec2_connection, region, eia_only):
    run_ec2_mxnet_inference(mxnet_inference_eia, RESNET_EIA_MODEL, "resnet-152-eia", ec2_connection, "eia", region, 80, 8081)


@pytest.mark.integration("elastic_inference")
@pytest.mark.model(RESNET_EIA_MODEL)
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_GPU_INSTANCE_TYPE, indirect=True)
@pytest.mark.parametrize("ei_accelerator_type", MX_EC2_EIA_ACCELERATOR_TYPE, indirect=True)
def test_ec2_mxnet_resnet_inference_eia_gpu(mxnet_inference_eia, ec2_connection, region, eia_only):
    run_ec2_mxnet_inference(mxnet_inference_eia, RESNET_EIA_MODEL, "resnet-152-eia", ec2_connection, "eia", region, 80, 8081)


@pytest.mark.integration("gluonnlp")
@pytest.mark.model(BERT_MODEL)
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_ec2_mxnet_gluonnlp_inference_cpu(mxnet_inference, ec2_connection, region, cpu_only, py3_only):
    run_ec2_mxnet_inference(mxnet_inference, BERT_MODEL, "gluonnlp", ec2_connection, "cpu", region, 90, 9091)


def run_ec2_mxnet_inference(image_uri, model_name, container_tag, ec2_connection, processor, region, target_port, target_management_port):
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
        ec2_connection.run(
            f"$(aws ecr get-login --no-include-email --region {region})", hide=True
        )
        LOGGER.info(docker_run_cmd)
        ec2_connection.run(docker_run_cmd, hide=True)
        if model_name == SQUEEZENET_MODEL:
            inference_result = test_utils.request_mxnet_inference(
                port=target_port, connection=ec2_connection, model="squeezenet"
            )
        elif model_name == BERT_MODEL:
            inference_result = test_utils.request_mxnet_inference_gluonnlp(
                port=target_port, connection=ec2_connection
            )
        elif model_name == RESNET_EIA_MODEL:
            inference_result = test_utils.request_mxnet_inference(
                port=target_port, connection=ec2_connection, model="resnet-152-eia"
            )
        assert (
            inference_result
        ), f"Failed to perform mxnet {model_name} inference test for image: {image_uri} on ec2"

    finally:
        ec2_connection.run(f"docker rm -f {container_name}", warn=True, hide=True)


@pytest.mark.integration("telemetry")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_mxnet_inference_telemetry_gpu(mxnet_inference, ec2_connection, gpu_only):
    execute_ec2_inference_test(ec2_connection, mxnet_inference, MX_TELEMETRY_CMD)


@pytest.mark.integration("telemetry")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_mxnet_inference_telemetry_cpu(mxnet_inference, ec2_connection, cpu_only):
    execute_ec2_inference_test(ec2_connection, mxnet_inference, MX_TELEMETRY_CMD)
