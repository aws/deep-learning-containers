import os

import pytest

from test.test_utils import CONTAINER_TESTS_PREFIX
from test.test_utils.ec2 import execute_ec2_training_test


PT_STANDALONE_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchStandalone")
PT_MNIST_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorch")
PT_REGRESSION_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchRegression")
PT_DGL_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "dgl_tests", "testPyTorchDGL")
PT_SMDEBUG_CMD = f"'{os.path.join(CONTAINER_TESTS_PREFIX, 'testSmdebug')} 'pytorch''"

PT_EC2_GPU_INSTANCE_TYPE = "p3.2xlarge"
PT_EC2_CPU_INSTANCE_TYPE = "c5.9xlarge"


@pytest.mark.parametrize("ec2_instance_type", [PT_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_pytorch_standalone_gpu(pytorch_training, ec2_connection, gpu_only):
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_STANDALONE_CMD)


@pytest.mark.parametrize("ec2_instance_type", [PT_EC2_CPU_INSTANCE_TYPE], indirect=True)
def test_pytorch_standalone_cpu(pytorch_training, ec2_connection, cpu_only):
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_STANDALONE_CMD)


@pytest.mark.parametrize("ec2_instance_type", [PT_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_pytorch_train_mnist_gpu(pytorch_training, ec2_connection, gpu_only):
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_MNIST_CMD)


@pytest.mark.skip(reason="Timing out, making sure others succeed")
@pytest.mark.parametrize("ec2_instance_type", [PT_EC2_CPU_INSTANCE_TYPE], indirect=True)
def test_pytorch_train_mnist_cpu(pytorch_training, ec2_connection, cpu_only):
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_MNIST_CMD)


@pytest.mark.parametrize("ec2_instance_type", [PT_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_pytorch_linear_regression_gpu(pytorch_training, ec2_connection, gpu_only):
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_REGRESSION_CMD)


@pytest.mark.parametrize("ec2_instance_type", [PT_EC2_CPU_INSTANCE_TYPE], indirect=True)
def test_pytorch_linear_regression_cpu(pytorch_training, ec2_connection, cpu_only):
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_REGRESSION_CMD)


@pytest.mark.parametrize("ec2_instance_type", [PT_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_pytorch_train_dgl_gpu(pytorch_training, ec2_connection, gpu_only, py3_only):
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_DGL_CMD)


@pytest.mark.parametrize("ec2_instance_type", [PT_EC2_CPU_INSTANCE_TYPE], indirect=True)
def test_pytorch_train_dgl_cpu(pytorch_training, ec2_connection, cpu_only, py3_only):
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_DGL_CMD)


@pytest.mark.parametrize("ec2_instance_type", [PT_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_pytorch_with_horovod(pytorch_training, ec2_connection, gpu_only):
    test_cmd = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPTHVD")
    execute_ec2_training_test(ec2_connection, pytorch_training, test_cmd)


@pytest.mark.parametrize("ec2_instance_type", [PT_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_pytorch_gloo(pytorch_training, ec2_connection, gpu_only, py3_only):
    test_cmd = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchGloo")
    execute_ec2_training_test(ec2_connection, pytorch_training, test_cmd)


@pytest.mark.parametrize("ec2_instance_type", [PT_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_pytorch_nccl(pytorch_training, ec2_connection, gpu_only, py3_only):
    test_cmd = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchNccl")
    execute_ec2_training_test(ec2_connection, pytorch_training, test_cmd)


@pytest.mark.parametrize("ec2_instance_type", [PT_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_pytorch_mpi(pytorch_training, ec2_connection, gpu_only, py3_only):
    test_cmd = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchMpi")
    execute_ec2_training_test(ec2_connection, pytorch_training, test_cmd)


@pytest.mark.parametrize("ec2_instance_type", [PT_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_pytorch_smdebug_gpu(pytorch_training, ec2_connection, gpu_only, py3_only):
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_SMDEBUG_CMD)


@pytest.mark.parametrize("ec2_instance_type", [PT_EC2_CPU_INSTANCE_TYPE], indirect=True)
def test_pytorch_smdebug_cpu(pytorch_training, ec2_connection, cpu_only, py3_only):
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_SMDEBUG_CMD)
