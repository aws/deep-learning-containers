import os

import pytest

from test import test_utils
from test.test_utils import CONTAINER_TESTS_PREFIX
from test.test_utils.ec2 import get_ec2_instance_type, execute_ec2_inference_test, get_ec2_accelerator_type
from test.dlc_tests.conftest import LOGGER


PT_EC2_GPU_INSTANCE_TYPE = get_ec2_instance_type(default="g3.8xlarge", processor="gpu")
PT_EC2_CPU_INSTANCE_TYPE = get_ec2_instance_type(default="c5.9xlarge", processor="cpu")
PT_EC2_EIA_ACCELERATOR_TYPE = get_ec2_accelerator_type(default="eia1.large", processor="eia")
PT_TELEMETRY_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "test_pt_dlc_telemetry_test")
PT_EC2_NEURON_ACCELERATOR_TYPE = get_ec2_accelerator_type(default="inf1.xlarge", processor="neuron")

@pytest.mark.model("resnet")
@pytest.mark.parametrize("ec2_instance_ami", [test_utils.NEURON_AL2_DLAMI], indirect=True)
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_NEURON_ACCELERATOR_TYPE, indirect=True)
def test_ec2_pytorch_inference_gpu(pytorch_inference, ec2_connection, region, gpu_only):
    ec2_pytorch_inference(pytorch_inference, "neuron", ec2_connection, region)

@pytest.mark.model("densenet")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_ec2_pytorch_inference_gpu(pytorch_inference, ec2_connection, region, gpu_only):
    ec2_pytorch_inference(pytorch_inference, "gpu", ec2_connection, region)


@pytest.mark.model("densenet")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_ec2_pytorch_inference_cpu(pytorch_inference, ec2_connection, region, cpu_only):
    ec2_pytorch_inference(pytorch_inference, "cpu", ec2_connection, region)


@pytest.mark.integration("elastic_inference")
@pytest.mark.model("resnet")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_CPU_INSTANCE_TYPE, indirect=True)
@pytest.mark.parametrize("ei_accelerator_type", PT_EC2_EIA_ACCELERATOR_TYPE, indirect=True)
def test_ec2_pytorch_inference_eia_cpu(pytorch_inference_eia, ec2_connection, region, eia_only):
    ec2_pytorch_inference(pytorch_inference_eia, "eia", ec2_connection, region)


@pytest.mark.integration("elastic_inference")
@pytest.mark.model("resnet")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INSTANCE_TYPE, indirect=True)
@pytest.mark.parametrize("ei_accelerator_type", PT_EC2_EIA_ACCELERATOR_TYPE, indirect=True)
@pytest.mark.skipif(PT_EC2_GPU_INSTANCE_TYPE == ["p3dn.24xlarge"], reason="Skipping EIA test on p3dn instances")
def test_ec2_pytorch_inference_eia_gpu(pytorch_inference_eia, ec2_connection, region, eia_only):
    ec2_pytorch_inference(pytorch_inference_eia, "eia", ec2_connection, region)


def ec2_pytorch_inference(image_uri, processor, ec2_connection, region):
    repo_name, image_tag = image_uri.split("/")[-1].split(":")
    container_name = f"{repo_name}-{image_tag}-ec2"
    model_name = "pytorch-densenet"
    if processor is "neuron":
        model_name = "pytorch-resnet-neuron"
    inference_cmd = test_utils.get_inference_run_command(image_uri, model_name, processor)
    docker_cmd = "nvidia-docker" if "gpu" in image_uri else "docker"

    if processor is "neuron":
        docker_run_cmd = (
            f"{docker_cmd} run -itd --name {container_name}"
            f" -p 80:8080 -p 8081:8081"
            f" --device=/dev/neuron0 --cap-add IPC_LOCK"
            f" {image_uri} {inference_cmd}"
        )
    else:
        docker_run_cmd = (
            f"{docker_cmd} run -itd --name {container_name}"
            f" -p 80:8080 -p 8081:8081"
            f" {image_uri} {inference_cmd}"
        )
    try:
        ec2_connection.run(
            f"$(aws ecr get-login --no-include-email --region {region})", hide=True
        )
        LOGGER.info(docker_run_cmd)
        ec2_connection.run(docker_run_cmd, hide=True)
        inference_result = test_utils.request_pytorch_inference_densenet(
            connection=ec2_connection
        )
        assert (
            inference_result
        ), f"Failed to perform pytorch inference test for image: {image_uri} on ec2"

    finally:
        ec2_connection.run(f"docker rm -f {container_name}", warn=True, hide=True)


@pytest.mark.integration("telemetry")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", ["p2.xlarge"], indirect=True)
def test_pytorch_inference_telemetry_gpu(pytorch_inference, ec2_connection, gpu_only):
    execute_ec2_inference_test(ec2_connection, pytorch_inference, PT_TELEMETRY_CMD)


@pytest.mark.integration("telemetry")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", ["c5.4xlarge"], indirect=True)
def test_pytorch_inference_telemetry_cpu(pytorch_inference, ec2_connection, cpu_only):
    execute_ec2_inference_test(ec2_connection, pytorch_inference, PT_TELEMETRY_CMD)
