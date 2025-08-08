import os

import pytest

import test.test_utils as test_utils
import test.test_utils.ec2 as ec2_utils

from test.test_utils import CONTAINER_TESTS_PREFIX, is_tf_version
from test.test_utils.ec2 import execute_ec2_training_test, get_ec2_instance_type

CURAND_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testCurand")
CURAND_EC2_SINGLE_GPU_INSTANCE_TYPE = get_ec2_instance_type(
    default="g5.8xlarge",
    processor="gpu",
    filter_function=ec2_utils.filter_only_single_gpu,
)


@pytest.mark.skip_serialized_release_pt_test
@pytest.mark.integration("curand")
@pytest.mark.model("N/A")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", CURAND_EC2_SINGLE_GPU_INSTANCE_TYPE, indirect=True)
def test_curand_gpu(training, ec2_connection, gpu_only, ec2_instance_type):
    if test_utils.is_image_incompatible_with_instance_type(training, ec2_instance_type):
        pytest.skip(f"Image {training} is incompatible with instance type {ec2_instance_type}")
    if is_tf_version("1", training) or "mxnet" in training or "arm64" in training:
        pytest.skip("Test is not configured for TF1 and MXNet and ARM64")
    execute_ec2_training_test(ec2_connection, training, CURAND_CMD)
