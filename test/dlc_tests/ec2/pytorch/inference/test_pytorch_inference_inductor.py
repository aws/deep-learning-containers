import os

from packaging.version import Version
from packaging.specifiers import SpecifierSet
import pytest

from test import test_utils
from test.test_utils import (
    get_framework_and_version_from_tag,
    get_inference_server_type,
    UL20_CPU_ARM64_US_WEST_2,
    login_to_ecr_registry,
    get_account_id_from_image_uri,
)
from test.test_utils.ec2 import get_ec2_instance_type, is_mainline_context
from test.dlc_tests.conftest import LOGGER


PT_EC2_CPU_INSTANCE_TYPE = get_ec2_instance_type(default="c5.9xlarge", processor="cpu")
PT_EC2_CPU_GRAVITON_INSTANCE_TYPES = ["c6g.4xlarge", "c7g.4xlarge"]
PT_EC2_GPU_GRAVITON_INSTANCE_TYPE = get_ec2_instance_type(
    default="g5g.4xlarge", processor="gpu", arch_type="graviton"
)
PT_EC2_CPU_ARM64_INSTANCE_TYPES = ["c6g.4xlarge", "c7g.4xlarge"]
PT_EC2_GPU_ARM64_INSTANCE_TYPE = get_ec2_instance_type(
    default="g5g.4xlarge", processor="gpu", arch_type="arm64"
)

PT_EC2_SINGLE_GPU_INSTANCE_TYPES = ["p3.2xlarge", "g4dn.4xlarge", "g5.4xlarge"]


@pytest.mark.model("densenet")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_SINGLE_GPU_INSTANCE_TYPES, indirect=True)
@pytest.mark.team("training-compiler")
@pytest.mark.skipif(
    is_mainline_context() and os.getenv("EC2_GPU_INSTANCE_TYPE") != "g4dn.xlarge",
    reason="Enforce test deduplication by running only alongside g4dn.xlarge tests.",
)
def test_ec2_pytorch_inference_gpu_inductor(
    pytorch_inference, ec2_connection, region, gpu_only, ec2_instance_type
):
    if test_utils.is_image_incompatible_with_instance_type(pytorch_inference, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_inference} is incompatible with instance type {ec2_instance_type}"
        )
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_inference)
    if Version(image_framework_version) in SpecifierSet("<2.0"):
        pytest.skip("skip the test as torch.compile only supported after 2.0")
    ec2_pytorch_inference(pytorch_inference, "gpu", ec2_connection, region)


@pytest.mark.model("densenet")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_CPU_INSTANCE_TYPE, indirect=True)
@pytest.mark.team("training-compiler")
def test_ec2_pytorch_inference_cpu_compilation(pytorch_inference, ec2_connection, region, cpu_only):
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_inference)
    if Version(image_framework_version) in SpecifierSet("<2.0"):
        pytest.skip("skip the test as torch.compile only supported after 2.0")
    ec2_pytorch_inference(pytorch_inference, "cpu", ec2_connection, region)


@pytest.mark.model("densenet")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_CPU_GRAVITON_INSTANCE_TYPES, indirect=True)
@pytest.mark.parametrize("ec2_instance_ami", [UL20_CPU_ARM64_US_WEST_2], indirect=True)
@pytest.mark.team("training-compiler")
@pytest.mark.skipif(
    is_mainline_context() and os.getenv("EC2_CPU_GRAVITON_INSTANCE_TYPE") != "c6g.4xlarge",
    reason="Enforce test deduplication by running only alongside c6g.4xlarge tests.",
)
def test_ec2_pytorch_inference_graviton_compilation_cpu(
    pytorch_inference_graviton, ec2_connection, region, cpu_only
):
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_inference_graviton)
    if Version(image_framework_version) in SpecifierSet("<2.0"):
        pytest.skip("skip the test as torch.compile only supported after 2.0")
    if "graviton" not in pytorch_inference_graviton:
        pytest.skip("skip EC2 tests for inductor")
    ec2_pytorch_inference(pytorch_inference_graviton, "cpu", ec2_connection, region)


@pytest.mark.model("densenet")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_CPU_ARM64_INSTANCE_TYPES, indirect=True)
@pytest.mark.parametrize("ec2_instance_ami", [UL20_CPU_ARM64_US_WEST_2], indirect=True)
@pytest.mark.team("training-compiler")
@pytest.mark.skipif(
    is_mainline_context() and os.getenv("EC2_CPU_ARM64_INSTANCE_TYPE") != "c6g.4xlarge",
    reason="Enforce test deduplication by running only alongside c6g.4xlarge tests.",
)
def test_ec2_pytorch_inference_arm64_compilation_cpu(
    pytorch_inference_arm64, ec2_connection, region, cpu_only
):
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_inference_arm64)
    if Version(image_framework_version) in SpecifierSet("<2.0"):
        pytest.skip("skip the test as torch.compile only supported after 2.0")
    if "arm64" not in pytorch_inference_arm64:
        pytest.skip("skip EC2 tests for inductor")
    ec2_pytorch_inference(pytorch_inference_arm64, "cpu", ec2_connection, region)


@pytest.mark.model("densenet")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_GRAVITON_INSTANCE_TYPE, indirect=True)
@pytest.mark.parametrize("ec2_instance_ami", [UL20_CPU_ARM64_US_WEST_2], indirect=True)
@pytest.mark.team("training-compiler")
def test_ec2_pytorch_inference_graviton_compilation_gpu(
    pytorch_inference_graviton, ec2_connection, region, gpu_only
):
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_inference_graviton)
    if Version(image_framework_version) in SpecifierSet("<2.0"):
        pytest.skip("skip the test as torch.compile only supported after 2.0")
    ec2_pytorch_inference(pytorch_inference_graviton, "gpu", ec2_connection, region)


@pytest.mark.model("densenet")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_ARM64_INSTANCE_TYPE, indirect=True)
@pytest.mark.parametrize("ec2_instance_ami", [UL20_CPU_ARM64_US_WEST_2], indirect=True)
@pytest.mark.team("training-compiler")
def test_ec2_pytorch_inference_arm64_compilation_gpu(
    pytorch_inference_arm64, ec2_connection, region, gpu_only
):
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_inference_arm64)
    if Version(image_framework_version) in SpecifierSet("<2.0"):
        pytest.skip("skip the test as torch.compile only supported after 2.0")
    ec2_pytorch_inference(pytorch_inference_arm64, "gpu", ec2_connection, region)


def ec2_pytorch_inference(image_uri, processor, ec2_connection, region):
    repo_name, image_tag = image_uri.split("/")[-1].split(":")
    container_name = f"{repo_name}-{image_tag}-ec2"
    model_name = "pytorch-densenet-inductor"

    inference_cmd = test_utils.get_inference_run_command(image_uri, model_name, processor)
    docker_runtime = "--runtime=nvidia --gpus all" if "gpu" in image_uri else ""

    docker_run_cmd = (
        f"docker run {docker_runtime} -itd --name {container_name}"
        f" -p 80:8080 -p 8081:8081"
        f" {image_uri} {inference_cmd}"
    )
    try:
        account_id = get_account_id_from_image_uri(image_uri)
        login_to_ecr_registry(ec2_connection, account_id, region)
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
