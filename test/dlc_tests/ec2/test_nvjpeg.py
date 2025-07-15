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


# @pytest.mark.usefixtures("sagemaker")
# @pytest.mark.model("N/A")
# @pytest.mark.parametrize("ec2_instance_type", ["g5g.2xlarge"], indirect=True)
# @pytest.mark.parametrize(
#     "ec2_instance_ami", [test_utils.AL2023_BASE_DLAMI_ARM64_US_WEST_2], indirect=True
# )
# @pytest.mark.timeout(1200)
# def test_nvjpeg_gpu_arm64(
#     gpu, ec2_connection, ec2_instance, arm64_compatible_only, below_cuda129_only
# ):
#     _run_nvjpeg_test(gpu, ec2_connection)


def _run_nvjpeg_test(image_uri, ec2_connection):
    """
    Runs the nvJPEG test on the specified image URI.
    """
    LOGGER.info(f"starting _run_nvjpeg_test with {image_uri}")

    account_id = test_utils.get_account_id_from_image_uri(image_uri)
    image_region = test_utils.get_region_from_image_uri(image_uri)
    repo_name, image_tag = test_utils.get_repository_and_tag_from_image_uri(image_uri)
    cuda_version = test_utils.get_cuda_version_from_tag(image_uri)

    container_name = f"{repo_name}-test-nvjpeg"

    LOGGER.info(f"_run_nvjpeg_test pulling: {image_uri}")
    test_utils.login_to_ecr_registry(ec2_connection, account_id, image_region)

    ec2_connection.run(f"docker pull {image_uri}", hide="out")

    LOGGER.info(f"_run_nvjpeg_test running: {image_uri}")
    ec2_connection.run(
        f"docker run --runtime=nvidia --gpus all --name {container_name} -id {image_uri}"
    )
    cuda_version_numeric = cuda_version.strip("cu")
    if cuda_version_numeric < Version("126"):
        # 12.4.1 has a different branch tag in cuda-samples
        if cuda_version_numeric == Version("124"):
            git_branch_tag = "12.4.1"
        else:
            git_branch_tag = f"{cuda_version_numeric[:-1]}.{cuda_version_numeric[-1]}"
        test_command = (
            f"git clone -b v{git_branch_tag} https://github.com/NVIDIA/cuda-samples.git && "
            "cd cuda-samples/Samples/4_CUDA_Libraries/nvJPEG && "
            "make -j$(nproc) && "
            "./nvJPEG"
        )
    else:
        # For CUDA 12.6 and above, we use the v12.8 branch of cuda-samples
        # This is a workaround for the issue where the nvJPEG sample in the
        # cuda-samples repository does not support compute_100 architecture.
        # The v12.8 branch is used to avoid the issue with compute_100 architecture.
        # See
    # sample 12.9 or master branch has compute_100 arch support issue
    # https://github.com/NVIDIA/cuda-samples/issues/367
        test_command = (
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
