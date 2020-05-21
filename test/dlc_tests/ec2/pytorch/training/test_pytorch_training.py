import os

import pytest

from test.test_utils import CONTAINER_TESTS_PREFIX, is_pr_context
from test.test_utils.ec2 import execute_ec2_training_test


PT_STANDALONE_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchStandalone")
PT_MNIST_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorch")
PT_REGRESSION_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchRegression")
PT_DGL_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "dgl_tests", "testPyTorchDGL")
PT_APEX_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testNVApex")
PT_TELEMETRY_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "test_pt_dlc_telemetry_test")

if is_pr_context():
    PT_EC2_GPU_INSTANCE_TYPE = ["p3.2xlarge"]
    PT_EC2_CPU_INSTANCE_TYPE = ["c5.9xlarge"]
else:
    # TODO: Add p3dn if releasing
    PT_EC2_GPU_INSTANCE_TYPE = ["g3.4xlarge", "p2.8xlarge", "p3.16xlarge"]
    PT_EC2_CPU_INSTANCE_TYPE = ["c4.8xlarge", "c5.18xlarge", "m4.16xlarge", "t2.2xlarge"]


@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_standalone_gpu(pytorch_training, ec2_connection, gpu_only):
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_STANDALONE_CMD)


@pytest.mark.parametrize("ec2_instance_type", PT_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_standalone_cpu(pytorch_training, ec2_connection, cpu_only):
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_STANDALONE_CMD)


@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_train_mnist_gpu(pytorch_training, ec2_connection, gpu_only):
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_MNIST_CMD)


@pytest.mark.skip(reason="Test is timing out, will assess in a different ticket")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_train_mnist_cpu(pytorch_training, ec2_connection, cpu_only):
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_MNIST_CMD)


@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_linear_regression_gpu(pytorch_training, ec2_connection, gpu_only):
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_REGRESSION_CMD)


@pytest.mark.parametrize("ec2_instance_type", PT_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_linear_regression_cpu(pytorch_training, ec2_connection, cpu_only):
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_REGRESSION_CMD)


@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_train_dgl_gpu(pytorch_training, ec2_connection, gpu_only, py3_only):
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_DGL_CMD)


@pytest.mark.parametrize("ec2_instance_type", PT_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_train_dgl_cpu(pytorch_training, ec2_connection, cpu_only, py3_only):
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_DGL_CMD)


@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_with_horovod(pytorch_training, ec2_connection, gpu_only):
    test_cmd = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPTHVD")
    execute_ec2_training_test(ec2_connection, pytorch_training, test_cmd)


@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_gloo(pytorch_training, ec2_connection, gpu_only, py3_only):
    test_cmd = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchGloo")
    execute_ec2_training_test(ec2_connection, pytorch_training, test_cmd)


@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_nccl(pytorch_training, ec2_connection, gpu_only, py3_only):
    test_cmd = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchNccl")
    execute_ec2_training_test(ec2_connection, pytorch_training, test_cmd)


@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_mpi(pytorch_training, ec2_connection, gpu_only, py3_only):
    test_cmd = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchMpi")
    execute_ec2_training_test(ec2_connection, pytorch_training, test_cmd)


@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_nvapex(pytorch_training, ec2_connection, gpu_only):
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_APEX_CMD)


@pytest.mark.parametrize("ec2_instance_type", ["p2.xlarge"], indirect=True)
def test_pytorch_telemetry_gpu(pytorch_training, ec2_connection, gpu_only):
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_TELEMETRY_CMD)


@pytest.mark.parametrize("ec2_instance_type", ["c5.4xlarge"], indirect=True)
def test_pytorch_telemetry_cpu(pytorch_training, ec2_connection, cpu_only):
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_TELEMETRY_CMD)
