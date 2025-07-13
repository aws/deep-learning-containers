import time

import pytest

from test import test_utils
from test.test_utils import ec2 as ec2_utils
from test.test_utils import LOGGER
from packaging.version import Version
from packaging.specifiers import SpecifierSet


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("N/A")
@pytest.mark.processor("gpu")
@pytest.mark.parametrize("ec2_instance_type", ["g5.8xlarge"], indirect=True)
@pytest.mark.timeout(1200)
def test_nvjpeg_gpu_x86(gpu, ec2_connection, ec2_instance, x86_compatible_only, below_cuda129_only):
    _run_nvjpeg_test(gpu, ec2_connection)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", ["g5g.2xlarge"], indirect=True)
@pytest.mark.parametrize(
    "ec2_instance_ami", [test_utils.AL2023_BASE_DLAMI_ARM64_US_WEST_2], indirect=True
)
@pytest.mark.timeout(1200)
def test_nvjpeg_gpu_arm64(
    gpu, ec2_connection, ec2_instance, arm64_compatible_only, below_cuda129_only
):
    _run_nvjpeg_test(gpu, ec2_connection)


def _run_nvjpeg_test(image_uri, ec2_connection):
    """
    Runs the nvJPEG test on the specified image URI.
    """
    LOGGER.info(f"starting _run_nvjpeg_test with {image_uri}")

    account_id = test_utils.get_account_id_from_image_uri(image_uri)
    image_region = test_utils.get_region_from_image_uri(image_uri)
    repo_name, image_tag = test_utils.get_repository_and_tag_from_image_uri(image_uri)

    container_name = f"{repo_name}-test-nvjpeg"

    LOGGER.info(f"_run_nvjpeg_test pulling: {image_uri}")
    test_utils.login_to_ecr_registry(ec2_connection, account_id, image_region)

    ec2_connection.run(f"docker pull {image_uri}", hide="out")

    LOGGER.info(f"_run_nvjpeg_test running: {image_uri}")
    ec2_connection.run(
        f"docker run --runtime=nvidia --gpus all --name {container_name} -id {image_uri}"
    )

    # sample 12.9 or master branch has compute_100 arch support issue
    # https://github.com/NVIDIA/cuda-samples/issues/367
    test_command_cu128 = (
        f"git clone -b v12.8 https://github.com/NVIDIA/cuda-samples.git && "
        "cd cuda-samples && "
        "mkdir build && cd build && "
        "cmake .. && "
        "cd Samples/4_CUDA_Libraries/nvJPEG && "
        "make -j$(nproc) && "
        "./nvJPEG"
    )

    output = ec2_connection.run(
        f"docker exec {container_name} /bin/bash -c '{test_command}'"
    ).stdout.strip("\n")

    return output
