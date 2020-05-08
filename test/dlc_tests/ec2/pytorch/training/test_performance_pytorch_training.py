import os

import pytest

from test.test_utils import CONTAINER_TESTS_PREFIX
from test.test_utils.ec2 import execute_ec2_training_performance_test


PT_PERFORMANCE_TRAINING_GPU_SYNTHETIC_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "performance_tests", "run_pytorch_training_performance_gpu_synthetic")

PT_EC2_GPU_SYNTHETIC_INSTANCE_TYPE = "p3.16xlarge"

@pytest.mark.skip(reason="Skip benchmark tests")
@pytest.mark.parametrize("ec2_instance_type", [PT_EC2_GPU_SYNTHETIC_INSTANCE_TYPE], indirect=True)
def test_performance_pytorch_gpu_synthetic(pytorch_training, ec2_connection, gpu_only, py3_only):
    execute_ec2_training_performance_test(ec2_connection, pytorch_training, PT_PERFORMANCE_TRAINING_GPU_SYNTHETIC_CMD)
