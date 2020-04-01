import os

import pytest

from test.test_utils import CONTAINER_TESTS_PREFIX
from test.test_utils.ec2 import execute_ec2_training_test


MX_STANDALONE_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testMXNetStandalone")
MX_MNIST_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testMXNet")
MX_DGL_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "dgl_tests", "testMXNetDGL")
MX_NLP_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "gluonnlp_tests", "testNLP")
MX_HVD_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testMXNetHVD")
MX_KERAS_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testKerasMXNet")

MX_SMDEBUG_CMD = f""" "{os.path.join(CONTAINER_TESTS_PREFIX, 'testSmdebug')} mxnet" """

MX_EC2_GPU_INSTANCE_TYPE = "p2.xlarge"
MX_EC2_CPU_INSTANCE_TYPE = "c5.4xlarge"


@pytest.mark.parametrize("ec2_instance_type", [MX_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_mxnet_standalone_gpu(mxnet_training, ec2_connection, gpu_only):
    execute_ec2_training_test(ec2_connection, mxnet_training, MX_STANDALONE_CMD)


@pytest.mark.parametrize("ec2_instance_type", [MX_EC2_CPU_INSTANCE_TYPE], indirect=True)
def test_mxnet_standalone_cpu(mxnet_training, ec2_connection, cpu_only):
    execute_ec2_training_test(ec2_connection, mxnet_training, MX_STANDALONE_CMD)


@pytest.mark.parametrize("ec2_instance_type", [MX_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_mxnet_train_mnist_gpu(mxnet_training, ec2_connection, gpu_only):
    execute_ec2_training_test(ec2_connection, mxnet_training, MX_MNIST_CMD)


@pytest.mark.parametrize("ec2_instance_type", [MX_EC2_CPU_INSTANCE_TYPE], indirect=True)
def test_mxnet_train_mnist_cpu(mxnet_training, ec2_connection, cpu_only):
    execute_ec2_training_test(ec2_connection, mxnet_training, MX_MNIST_CMD)


@pytest.mark.parametrize("ec2_instance_type", [MX_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_mxnet_keras_gpu(mxnet_training, ec2_connection, gpu_only):
    execute_ec2_training_test(ec2_connection, mxnet_training, MX_KERAS_CMD)


@pytest.mark.skip(reason="Test is timing out, will assess in a different ticket")
@pytest.mark.parametrize("ec2_instance_type", [MX_EC2_CPU_INSTANCE_TYPE], indirect=True)
def test_mxnet_keras_cpu(mxnet_training, ec2_connection, cpu_only):
    execute_ec2_training_test(ec2_connection, mxnet_training, MX_KERAS_CMD)


@pytest.mark.parametrize("ec2_instance_type", [MX_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_mxnet_train_dgl_gpu(mxnet_training, ec2_connection, gpu_only, py3_only):
    execute_ec2_training_test(ec2_connection, mxnet_training, MX_DGL_CMD)


@pytest.mark.parametrize("ec2_instance_type", [MX_EC2_CPU_INSTANCE_TYPE], indirect=True)
def test_mxnet_train_dgl_cpu(mxnet_training, ec2_connection, cpu_only, py3_only):
    execute_ec2_training_test(ec2_connection, mxnet_training, MX_DGL_CMD)


@pytest.mark.parametrize("ec2_instance_type", [MX_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_mxnet_train_nlp_gpu(mxnet_training, ec2_connection, gpu_only, py3_only):
    execute_ec2_training_test(ec2_connection, mxnet_training, MX_NLP_CMD)


@pytest.mark.parametrize("ec2_instance_type", [MX_EC2_CPU_INSTANCE_TYPE], indirect=True)
def test_mxnet_train_nlp_cpu(mxnet_training, ec2_connection, cpu_only, py3_only):
    execute_ec2_training_test(ec2_connection, mxnet_training, MX_NLP_CMD)


@pytest.mark.parametrize("ec2_instance_type", [MX_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_mxnet_with_horovod_gpu(mxnet_training, ec2_connection, gpu_only):
    execute_ec2_training_test(ec2_connection, mxnet_training, MX_HVD_CMD)


@pytest.mark.parametrize("ec2_instance_type", [MX_EC2_CPU_INSTANCE_TYPE], indirect=True)
def test_mxnet_with_horovod_cpu(mxnet_training, ec2_connection, cpu_only):
    execute_ec2_training_test(ec2_connection, mxnet_training, MX_HVD_CMD)


@pytest.mark.skip(reason="Test is not properly receiving args. Will assess in a different ticket.")
@pytest.mark.parametrize("ec2_instance_type", [MX_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_mxnet_smdebug_gpu(mxnet_training, ec2_connection, gpu_only, py3_only):
    execute_ec2_training_test(ec2_connection, mxnet_training, MX_SMDEBUG_CMD)


@pytest.mark.skip(reason="Test is not properly receiving args. Will assess in a different ticket.")
@pytest.mark.parametrize("ec2_instance_type", [MX_EC2_CPU_INSTANCE_TYPE], indirect=True)
def test_mxnet_smdebug_cpu(mxnet_training, ec2_connection, cpu_only, py3_only):
    execute_ec2_training_test(ec2_connection, mxnet_training, MX_SMDEBUG_CMD)
