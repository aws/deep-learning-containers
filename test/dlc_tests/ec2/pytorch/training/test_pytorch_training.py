import os

from test.test_utils import CONTAINER_TESTS_PREFIX
from test.test_utils.ec2 import ec2_training_test_executor


def test_pytorch_standalone(pytorch_training):
    test_script = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchStandalone")
    ec2_training_test_executor(pytorch_training, test_script)


# TODO: REMOVE CPU ONLY WHEN GPU TESTS ARE ENABLED
def test_pytorch_train_mnist(pytorch_training, cpu_only):
    test_script = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorch")
    ec2_training_test_executor(pytorch_training, test_script)


# TODO: REMOVE CPU ONLY WHEN GPU TESTS ARE ENABLED
def test_pytorch_linear_regression(pytorch_training, cpu_only):
    test_script = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchRegression")
    ec2_training_test_executor(pytorch_training, test_script)


# TODO: REMOVE CPU ONLY WHEN GPU TESTS ARE ENABLED
def test_pytorch_train_dgl(pytorch_training, py3_only, cpu_only):
    test_script = os.path.join(CONTAINER_TESTS_PREFIX, "dgl_tests", "testPyTorchDGL")
    ec2_training_test_executor(pytorch_training, test_script)


def test_pytorch_with_horovod(pytorch_training, gpu_only):
    test_script = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPTHVD")
    ec2_training_test_executor(pytorch_training, test_script)


def test_pytorch_gloo(pytorch_training, gpu_only, py3_only):
    test_script = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchGloo")
    ec2_training_test_executor(pytorch_training, test_script)


def test_pytorch_nccl(pytorch_training, gpu_only, py3_only):
    test_script = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchNccl")
    ec2_training_test_executor(pytorch_training, test_script)


def test_pytorch_mpi(pytorch_training, gpu_only, py3_only):
    """
    Only supported with Horovod
    """
    test_script = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchMpi")
    ec2_training_test_executor(pytorch_training, test_script)


def test_smdebug(pytorch_training, py3_only):
    test_script = os.path.join(CONTAINER_TESTS_PREFIX, "testSmdebug")
    test_cmd = f"{test_script} mxnet"
    ec2_training_test_executor(pytorch_training, test_cmd)
