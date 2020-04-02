import os

import pytest

from test.test_utils import CONTAINER_TESTS_PREFIX
from test.test_utils.ec2 import execute_ec2_training_test


TF1_STANDALONE_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testTensorflow1Standalone")
TF2_STANDALONE_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testTensorflow2Standalone")
TF_MNIST_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testTensorFlow")
TF_HVD_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testHVD")

TF_EC2_GPU_INSTANCE_TYPE = "p2.xlarge"
TF_EC2_CPU_INSTANCE_TYPE = "c5.4xlarge"


@pytest.mark.parametrize("ec2_instance_type", [TF_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_tensorflow_standalone_gpu(tensorflow_training, ec2_connection, gpu_only):
    test_script = TF1_STANDALONE_CMD if "1." in tensorflow_training else TF2_STANDALONE_CMD
    execute_ec2_training_test(ec2_connection, tensorflow_training, test_script)


@pytest.mark.parametrize("ec2_instance_type", [TF_EC2_CPU_INSTANCE_TYPE], indirect=True)
def test_tensorflow_standalone_cpu(tensorflow_training, ec2_connection, cpu_only):
    test_script = TF1_STANDALONE_CMD if "1." in tensorflow_training else TF2_STANDALONE_CMD
    execute_ec2_training_test(ec2_connection, tensorflow_training, test_script)


@pytest.mark.parametrize("ec2_instance_type", [TF_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_tensorflow_train_mnist_gpu(tensorflow_training, ec2_connection, gpu_only):
    execute_ec2_training_test(ec2_connection, tensorflow_training, TF_MNIST_CMD)


@pytest.mark.parametrize("ec2_instance_type", [TF_EC2_CPU_INSTANCE_TYPE], indirect=True)
def test_tensorflow_train_mnist_cpu(tensorflow_training, ec2_connection, cpu_only):
    execute_ec2_training_test(ec2_connection, tensorflow_training, TF_MNIST_CMD)


@pytest.mark.parametrize("ec2_instance_type", [TF_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_tensorflow_with_horovod_gpu(tensorflow_training, ec2_connection, gpu_only):
    execute_ec2_training_test(ec2_connection, tensorflow_training, TF_HVD_CMD)


@pytest.mark.parametrize("ec2_instance_type", [TF_EC2_CPU_INSTANCE_TYPE], indirect=True)
def test_tensorflow_with_horovod_cpu(tensorflow_training, ec2_connection, cpu_only):
    execute_ec2_training_test(ec2_connection, tensorflow_training, TF_HVD_CMD)
