import os

import pytest

from test.test_utils import CONTAINER_TESTS_PREFIX


CONTAINER_TEST_LOCAL_DIR = os.path.join(os.path.expanduser('~'), 'container_tests')


@pytest.mark.parametrize("ec2_instance_type", ["p3.8xlarge"], indirect=True)
def test_pytorch_standalone_gpu(pytorch_training, ec2_connection, gpu_only):
    conn = ec2_connection

    conn.run(f"docker run -v {CONTAINER_TEST_LOCAL_DIR}:{os.path.join(os.sep, 'test')} {pytorch_training} "
             f"{os.path.join(os.sep, 'bin', 'bash')} -c "
             f"{os.path.join(CONTAINER_TESTS_PREFIX, 'pytorch_tests', 'testPyTorchStandalone')}")


def test_pytorch_standalone_cpu(pytorch_training, ec2_connection, cpu_only):
    print(pytorch_training)


def test_pytorch_train_mnist_gpu(pytorch_training, ec2_connection, gpu_only):
    print(pytorch_training)


def test_pytorch_train_mnist_cpu(pytorch_training, ec2_connection, cpu_only):
    print(pytorch_training)


def test_pytorch_linear_regression_gpu(pytorch_training, ec2_connection, gpu_only):
    print(pytorch_training)


def test_pytorch_linear_regression_cpu(pytorch_training, ec2_connection, cpu_only):
    print(pytorch_training)


def test_pytorch_train_dgl_gpu(pytorch_training, ec2_connection, gpu_only, py3_only):
    print(pytorch_training)


def test_pytorch_train_dgl_cpu(pytorch_training, ec2_connection, cpu_only, py3_only):
    print(pytorch_training)


def test_pytorch_with_horovod(pytorch_training, ec2_connection, gpu_only):
    print(pytorch_training)


def test_pytorch_gloo(pytorch_training, ec2_connection, gpu_only, py3_only):
    print(pytorch_training)


def test_pytorch_nccl(pytorch_training, ec2_connection, gpu_only, py3_only):
    print(pytorch_training)


def test_pytorch_mpi(pytorch_training, ec2_connection, gpu_only, py3_only):
    print(pytorch_training)


def test_pytorch_smdebug_gpu(pytorch_training, ec2_connection, gpu_only, py3_only):
    print(pytorch_training)


def test_pytorch_smdebug_cpu(pytorch_training, ec2_connection, cpu_only, py3_only):
    print(pytorch_training)
