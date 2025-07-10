import time

import pytest

from test import test_utils
from test.test_utils import ec2 as ec2_utils
from test.test_utils import LOGGER
from packaging.version import Version
from packaging.specifiers import SpecifierSet


@pytest.mark.usefixtures("sagemaker", "huggingface", "telemetry")
@pytest.mark.model("N/A")
@pytest.mark.processor("gpu")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["g5.8xlarge"], indirect=True)
@pytest.mark.timeout(1200)
def test_nvjpeg_gpu(gpu, ec2_connection, x86_compatible_only, below_cuda129_only):
    _run_nvjpeg_test(gpu, ec2_connection)


@pytest.mark.usefixtures("sagemaker", "telemetry")
@pytest.mark.model("N/A")
@pytest.mark.processor("gpu")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["g5g.2xlarge"], indirect=True)
@pytest.mark.parametrize(
    "ec2_instance_ami", [test_utils.AL2023_BASE_DLAMI_ARM64_US_WEST_2], indirect=True
)
@pytest.mark.timeout(1200)
def test_nvjpeg_gpu(gpu, ec2_connection, arm64_compatible_only, below_cuda129_only):
    _run_nvjpeg_test(gpu, ec2_connection)


def _run_nvjpeg_test(image_uri, ec2_connection):
    """
    Runs the nvJPEG test on the specified image URI.
    """
    LOGGER.info(f"starting _run_nvjpeg_test with {image_uri}")

    account_id = test_utils.get_account_id_from_image_uri(image_uri)
    image_region = test_utils.get_region_from_image_uri(image_uri)
    repo_name, image_tag = test_utils.get_repository_and_tag_from_image_uri(image_uri)
    cuda_version = test_utils.get_cuda_version_from_tag(image_uri)
    # convert cu126 to 12.6
    numbers = cuda_version[2:]
    numeric_version = f"{numbers[:-1]}.{numbers[-1]}"

    test_repo_cuda_tag = numeric_version
    # test repo only has tags v12.1-v12.4, v12.5, v12.8, v12.9
    if numeric_version in ["12.6", "12.7"]:
        test_repo_cuda_tag = "12.5"

    container_name = f"{repo_name}-test-nvjpeg"

    LOGGER.info(f"_run_nvjpeg_test pulling: {image_uri}")
    test_utils.login_to_ecr_registry(ec2_connection, account_id, image_region)

    ec2_connection.run(f"docker pull {image_uri}", hide="out")

    LOGGER.info(f"_run_nvjpeg_test running: {image_uri}")
    ec2_connection.run(
        f"docker run --runtime=nvidia --gpus all --name {container_name} -id {image_uri}"
    )
    test_command = (
        f"git clone -b v{test_repo_cuda_tag} https://github.com/NVIDIA/cuda-samples.git && "
        "cd cuda-samples/Samples/4_CUDA_Libraries/nvJPEG && "
        "make -j $(nproc) && "
        "./nvjpeg"
    )

    output = ec2_connection.run(
        f"docker exec {container_name} /bin/bash -c '{test_command}'"
    ).stdout.strip("\n")

    return output
