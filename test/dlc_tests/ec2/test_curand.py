import os

import pytest

from test.test_utils import CONTAINER_TESTS_PREFIX, is_tf1
from test.test_utils.ec2 import execute_ec2_training_test

PT_CURAND_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testCurand")


@pytest.mark.integration("curand")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", ["p2.xlarge"], indirect=True)
def test_curand_gpu(training, ec2_connection, gpu_only):
    if is_tf1(training) or "mxnet" in training:
        pytest.skip("Test is not configured for TF1 and MXNet")
    execute_ec2_training_test(ec2_connection, training, PT_CURAND_CMD)