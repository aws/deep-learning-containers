import os

import pytest

from test.test_utils import CONTAINER_TESTS_PREFIX
from test.test_utils.ec2 import execute_ec2_training_performance_test


TF_PERFORMANCE_TRAINING_CPU_SYNTHETIC_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "benchmark", "run_tensorflow_training_performance_cpu")
TF_PERFORMANCE_TRAINING_GPU_SYNTHETIC_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "benchmark", "run_tensorflow_training_performance_gpu_synthetic")
TF_PERFORMANCE_TRAINING_GPU_IMAGENET_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "benchmark", "run_tensorflow_training_performance_gpu_imagenet")

TF_EC2_GPU_INSTANCE_TYPE = "p3.16xlarge"
TF_EC2_CPU_INSTANCE_TYPE = "c5.18xlarge"

@pytest.mark.skip()
@pytest.mark.integration("synthetic dataset")
@pytest.mark.model("resnet50")
@pytest.mark.parametrize("ec2_instance_type", [TF_EC2_CPU_INSTANCE_TYPE], indirect=True)
def test_performance_tensorflow_cpu(tensorflow_training, ec2_connection, cpu_only):
    execute_ec2_training_performance_test(ec2_connection, tensorflow_training, TF_PERFORMANCE_TRAINING_CPU_SYNTHETIC_CMD)

@pytest.mark.skip()
@pytest.mark.integration("synthetic dataset")
@pytest.mark.model("resnet50")
@pytest.mark.parametrize("ec2_instance_type", [TF_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_performance_tensorflow_gpu_synthetic(tensorflow_training, ec2_connection, gpu_only):
    execute_ec2_training_performance_test(ec2_connection, tensorflow_training, TF_PERFORMANCE_TRAINING_GPU_SYNTHETIC_CMD)


@pytest.mark.integration("imagenet dataset")
@pytest.mark.model("resnet50")
@pytest.mark.parametrize("ec2_instance_type", [TF_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_performance_tensorflow_gpu_imagenet(tensorflow_training, ec2_connection, gpu_only):
    execute_ec2_training_performance_test(ec2_connection, tensorflow_training, TF_PERFORMANCE_TRAINING_GPU_IMAGENET_CMD)
