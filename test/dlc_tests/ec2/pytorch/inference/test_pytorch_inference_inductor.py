from packaging.version import Version
from packaging.specifiers import SpecifierSet
import pytest

from test import test_utils
from test.test_utils import (
    CONTAINER_TESTS_PREFIX,
    get_framework_and_version_from_tag,
    get_inference_server_type,
    UL20_CPU_ARM64_US_WEST_2,
)
from test.test_utils.ec2 import (
    get_ec2_instance_type,
    execute_ec2_inference_test,
    get_ec2_accelerator_type,
)
from test.dlc_tests.conftest import LOGGER


PT_EC2_CPU_INSTANCE_TYPE = get_ec2_instance_type(default="c5.9xlarge", processor="cpu")
PT_EC2_GRAVITON_INSTANCE_TYPES = ["c6g.4xlarge", "c7g.4xlarge"]
PT_EC2_SINGLE_GPU_INSTANCE_TYPES = ["p3.2xlarge", "g4dn.4xlarge", "g5.4xlarge"]


@pytest.mark.model("densenet")
@pytest.mark.parametrize(
    "ec2_instance_type", PT_EC2_SINGLE_GPU_INSTANCE_TYPES, indirect=True
)
def test_ec2_pytorch_inference_gpu_inductor(
    pytorch_inference, ec2_connection, region, gpu_only, ec2_instance_type
):
    if test_utils.is_image_incompatible_with_instance_type(
        pytorch_inference, ec2_instance_type
    ):
        pytest.skip(
            f"Image {pytorch_inference} is incompatible with instance type {ec2_instance_type}"
        )
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_inference)
    if Version(image_framework_version) in SpecifierSet("<2.0"):
        pytest.skip("skip the test as torch.compile only supported after 2.0")
    ec2_pytorch_inference(pytorch_inference, "gpu", ec2_connection, region)


@pytest.mark.model("densenet")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_ec2_pytorch_inference_cpu_compilation(
    pytorch_inference, ec2_connection, region, cpu_only
):
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_inference)
    if Version(image_framework_version) in SpecifierSet("<2.0"):
        pytest.skip("skip the test as torch.compile only supported after 2.0")
    ec2_pytorch_inference(pytorch_inference, "cpu", ec2_connection, region)


@pytest.mark.model("densenet")
@pytest.mark.parametrize(
    "ec2_instance_type", PT_EC2_GRAVITON_INSTANCE_TYPES, indirect=True
)
@pytest.mark.parametrize("ec2_instance_ami", [UL20_CPU_ARM64_US_WEST_2], indirect=True)
def test_ec2_pytorch_inference_cpu_compilation(
    pytorch_inference_graviton, ec2_connection, region, cpu_only
):
    _, image_framework_version = get_framework_and_version_from_tag(
        pytorch_inference_graviton
    )
    if Version(image_framework_version) in SpecifierSet("<2.0"):
        pytest.skip("skip the test as torch.compile only supported after 2.0")
    if "graviton" not in pytorch_inference_graviton:
        pytest.skip("skip EC2 tests for inductor")
    ec2_pytorch_inference(
        pytorch_inference_graviton, "graviton", ec2_connection, region
    )


def ec2_pytorch_inference(image_uri, processor, ec2_connection, region):
    repo_name, image_tag = image_uri.split("/")[-1].split(":")
    container_name = f"{repo_name}-{image_tag}-ec2"
    model_name = "pytorch-densenet-inductor"

    inference_cmd = test_utils.get_inference_run_command(
        image_uri, model_name, processor
    )
    docker_cmd = "nvidia-docker" if "gpu" in image_uri else "docker"

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
        server_type = get_inference_server_type(image_uri)
        inference_result = test_utils.request_pytorch_inference_densenet(
            connection=ec2_connection, model_name=model_name, server_type=server_type
        )
        assert (
            inference_result
        ), f"Failed to perform pytorch inference test for image: {image_uri} on ec2"

    finally:
        ec2_connection.run(f"docker rm -f {container_name}", warn=True, hide=True)
