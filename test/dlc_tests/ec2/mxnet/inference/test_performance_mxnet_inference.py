import os

import pytest

from test.test_utils import CONTAINER_TESTS_PREFIX
from test.test_utils.ec2 import execute_ec2_training_performance_test


TF_PERFORMANCE_INFERENCE_GPU_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "performance_tests", "run_mxnet_inference_performance_gpu")
TF_PERFORMANCE_INFERENCE_CPU_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "performance_tests", "run_mxnet_inference_performance_gpu")

MX_EC2_GPU_INSTANCE_TYPE = "p3.16xlarge"
MX_EC2_CPU_INSTANCE_TYPE = "c5.18xlarge"


@pytest.mark.parametrize("ec2_instance_type", [MX_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_performance_mxnet_inference_gpu(mxnet_training, ec2_connection, gpu_only):
    execute_ec2_training_performance_test(ec2_connection, mxnet_training, TF_PERFORMANCE_INFERENCE_GPU_CMD) # will rename this function


@pytest.mark.parametrize("ec2_instance_type", [MX_EC2_CPU_INSTANCE_TYPE], indirect=True)
def test_performance_mxnet_inference_cpu(mxnet_training, ec2_connection, cpu_only):
    execute_ec2_training_performance_test(ec2_connection, mxnet_training, TF_PERFORMANCE_INFERENCE_CPU_CMD)
