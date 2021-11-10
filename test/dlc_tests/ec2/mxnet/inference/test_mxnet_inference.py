import os
import pytest

import test.test_utils.ec2 as ec2_utils

from test import test_utils
from test.test_utils import CONTAINER_TESTS_PREFIX, get_framework_and_version_from_tag
from test.test_utils.ec2 import get_ec2_instance_type, execute_ec2_inference_test, get_ec2_accelerator_type
from test.dlc_tests.conftest import LOGGER


SQUEEZENET_MODEL = "squeezenet"
BERT_MODEL = "bert_sst"
RESNET_EIA_MODEL = "resnet-152-eia"


MX_EC2_GPU_INSTANCE_TYPE = get_ec2_instance_type(default="g3.8xlarge", processor="gpu")
MX_EC2_CPU_INSTANCE_TYPE = get_ec2_instance_type(default="c5.4xlarge", processor="cpu")
MX_EC2_EIA_ACCELERATOR_TYPE = get_ec2_accelerator_type(default="eia1.large", processor="eia")
MX_EC2_GPU_EIA_INSTANCE_TYPE = get_ec2_instance_type(
    default="g3.8xlarge", processor="gpu", filter_function=ec2_utils.filter_not_heavy_instance_types,
)
MX_EC2_SINGLE_GPU_INSTANCE_TYPE = get_ec2_instance_type(
    default="p3.2xlarge", processor="gpu", filter_function=ec2_utils.filter_only_single_gpu,
)
MX_EC2_NEURON_INSTANCE_TYPE = get_ec2_instance_type(default="inf1.xlarge", processor="neuron")
MX_EC2_GRAVITON_INSTANCE_TYPE = get_ec2_instance_type(default="c6g.4xlarge", processor="cpu")

MX_TELEMETRY_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "test_mx_dlc_telemetry_test")

@pytest.mark.model("mxnet-resnet-neuron")
@pytest.mark.parametrize("ec2_instance_ami", [test_utils.NEURON_UBUNTU_18_BASE_DLAMI_US_WEST_2], indirect=True)
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_NEURON_INSTANCE_TYPE, indirect=True)
def test_ec2_mxnet_inference_neuron(mxnet_inference_neuron, ec2_connection, region):
    run_ec2_mxnet_inference(mxnet_inference_neuron, "mxnet-resnet-neuron", "resnet", ec2_connection, "neuron", region, 80, 8081)


@pytest.mark.model(SQUEEZENET_MODEL)
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_ec2_mxnet_squeezenet_inference_gpu(mxnet_inference, ec2_connection, region, gpu_only, ec2_instance_type):
    if test_utils.is_image_incompatible_with_instance_type(mxnet_inference, ec2_instance_type):
        pytest.skip(f"Image {mxnet_inference} is incompatible with instance type {ec2_instance_type}")
    run_ec2_mxnet_inference(mxnet_inference, SQUEEZENET_MODEL, "squeezenet", ec2_connection, "gpu", region, 80, 8081)


@pytest.mark.integration("gluonnlp")
@pytest.mark.model(BERT_MODEL)
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_ec2_mxnet_gluonnlp_inference_gpu(
        mxnet_inference, ec2_connection, region, gpu_only, py3_only, ec2_instance_type
):
    if test_utils.is_image_incompatible_with_instance_type(mxnet_inference, ec2_instance_type):
        pytest.skip(f"Image {mxnet_inference} is incompatible with instance type {ec2_instance_type}")
    run_ec2_mxnet_inference(mxnet_inference, BERT_MODEL, "gluonnlp", ec2_connection, "gpu", region, 90, 9091)


@pytest.mark.model(SQUEEZENET_MODEL)
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_ec2_mxnet_squeezenet_inference_cpu(mxnet_inference, ec2_connection, region, cpu_only):
    run_ec2_mxnet_inference(mxnet_inference, SQUEEZENET_MODEL, "squeezenet", ec2_connection, "cpu", region, 80, 8081)


@pytest.mark.integration("elastic_inference")
@pytest.mark.model(RESNET_EIA_MODEL)
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_CPU_INSTANCE_TYPE, indirect=True)
@pytest.mark.parametrize("ei_accelerator_type", MX_EC2_EIA_ACCELERATOR_TYPE, indirect=True)
def test_ec2_mxnet_resnet_inference_eia_cpu(mxnet_inference_eia, ec2_connection, region):
    model_name = RESNET_EIA_MODEL
    image_framework, image_framework_version = get_framework_and_version_from_tag(mxnet_inference_eia)
    if image_framework_version == "1.5.1":
        model_name = "resnet-152-eia-1-5-1"
    run_ec2_mxnet_inference(mxnet_inference_eia, model_name, "resnet-152-eia", ec2_connection, "eia", region, 80, 8081)


@pytest.mark.integration("elastic_inference")
@pytest.mark.model(RESNET_EIA_MODEL)
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_GPU_EIA_INSTANCE_TYPE, indirect=True)
@pytest.mark.parametrize("ei_accelerator_type", MX_EC2_EIA_ACCELERATOR_TYPE, indirect=True)
def test_ec2_mxnet_resnet_inferencei_eia_gpu(mxnet_inference_eia, ec2_connection, region):
    model_name = RESNET_EIA_MODEL
    image_framework, image_framework_version = get_framework_and_version_from_tag(mxnet_inference_eia)
    if image_framework_version == "1.5.1":
        model_name = "resnet-152-eia-1-5-1"
    run_ec2_mxnet_inference(mxnet_inference_eia, model_name, "resnet-152-eia", ec2_connection, "eia", region, 80, 8081)


@pytest.mark.model(SQUEEZENET_MODEL)
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_GRAVITON_INSTANCE_TYPE, indirect=True)
def test_ec2_mxnet_inference_graviton_cpu(mxnet_inference_graviton, ec2_connection, region):
    run_ec2_mxnet_inference(mxnet_inference_graviton, SQUEEZENET_MODEL, "squeezenet", ec2_connection, "cpu", region, 80, 8081)


@pytest.mark.integration("gluonnlp")
@pytest.mark.model(BERT_MODEL)
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_ec2_mxnet_gluonnlp_inference_cpu(mxnet_inference, ec2_connection, region, cpu_only, py3_only):
    run_ec2_mxnet_inference(mxnet_inference, BERT_MODEL, "gluonnlp", ec2_connection, "cpu", region, 90, 9091)


def run_ec2_mxnet_inference(image_uri, model_name, container_tag, ec2_connection, processor, region, target_port, target_management_port):
    repo_name, image_tag = image_uri.split("/")[-1].split(":")
    container_name = f"{repo_name}-{image_tag}-ec2-{container_tag}"
    docker_cmd = "nvidia-docker" if "gpu" in image_uri else "docker"
    mms_inference_cmd = test_utils.get_inference_run_command(image_uri, model_name, processor)
    if processor == "neuron":
        docker_run_cmd = (
            f"{docker_cmd} run -itd --name {container_name}"
            f" -p {target_port}:8080 -p {target_management_port}:8081"
            f" --device=/dev/neuron0 --cap-add IPC_LOCK"
            f" {image_uri} {mms_inference_cmd}"
        )
    else:
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
                port=target_port, connection=ec2_connection, model=model_name
            )
        elif model_name == "mxnet-resnet-neuron":
            inference_result = test_utils.request_mxnet_inference(
                port=target_port, connection=ec2_connection, model=model_name
            )
        assert (
            inference_result
        ), f"Failed to perform mxnet {model_name} inference test for image: {image_uri} on ec2"

    finally:
        ec2_connection.run(f"docker rm -f {container_name}", warn=True, hide=True)


@pytest.mark.flaky(reruns=3)
@pytest.mark.integration("telemetry")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_SINGLE_GPU_INSTANCE_TYPE, indirect=True)
def test_mxnet_inference_telemetry_gpu(mxnet_inference, ec2_connection, gpu_only, ec2_instance_type):
    if test_utils.is_image_incompatible_with_instance_type(mxnet_inference, ec2_instance_type):
        pytest.skip(f"Image {mxnet_inference} is incompatible with instance type {ec2_instance_type}")
    execute_ec2_inference_test(ec2_connection, mxnet_inference, MX_TELEMETRY_CMD)


@pytest.mark.flaky(reruns=3)
@pytest.mark.integration("telemetry")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_mxnet_inference_telemetry_cpu(mxnet_inference, ec2_connection, cpu_only):
    execute_ec2_inference_test(ec2_connection, mxnet_inference, MX_TELEMETRY_CMD)


@pytest.mark.flaky(reruns=3)
@pytest.mark.integration("telemetry")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_GRAVITON_INSTANCE_TYPE, indirect=True)
def test_mxnet_inference_telemetry_graviton_cpu(mxnet_inference_graviton, ec2_connection):
    execute_ec2_inference_test(ec2_connection, mxnet_inference_graviton, MX_TELEMETRY_CMD)
    