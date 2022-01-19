import os

import pytest

import test.test_utils as test_utils
import test.test_utils.ec2 as ec2_utils

from test.test_utils import CONTAINER_TESTS_PREFIX
from test.test_utils.ec2 import execute_ec2_training_test, get_ec2_instance_type

from packaging.version import Version

MX_STANDALONE_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testMXNetStandalone")
MX_MNIST_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testMXNet")
MX_DGL_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "dgl_tests", "testMXNetDGL")
MX_NLP_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "gluonnlp_tests", "testNLP")
MX_HVD_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testMXNetHVD")
MX_KERAS_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testKerasMXNet")
MX_TELEMETRY_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "test_mx_dlc_telemetry_test")

MX_EC2_GPU_INSTANCE_TYPE = get_ec2_instance_type(default="g3.8xlarge", processor="gpu")
MX_EC2_CPU_INSTANCE_TYPE = get_ec2_instance_type(default="c5.4xlarge", processor="cpu")
MX_EC2_SINGLE_GPU_INSTANCE_TYPE = get_ec2_instance_type(
    default="p3.2xlarge", processor="gpu", filter_function=ec2_utils.filter_only_single_gpu,
)


@pytest.mark.integration("mxnet_sanity_test")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_mxnet_standalone_gpu(mxnet_training, ec2_connection, gpu_only, ec2_instance_type):
    if test_utils.is_image_incompatible_with_instance_type(mxnet_training, ec2_instance_type):
        pytest.skip(f"Image {mxnet_training} is incompatible with instance type {ec2_instance_type}")
    execute_ec2_training_test(ec2_connection, mxnet_training, MX_STANDALONE_CMD)


@pytest.mark.integration("mxnet_sanity_test")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_mxnet_standalone_cpu(mxnet_training, ec2_connection, cpu_only):
    execute_ec2_training_test(ec2_connection, mxnet_training, MX_STANDALONE_CMD)


@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_mxnet_train_mnist_gpu(mxnet_training, ec2_connection, gpu_only, ec2_instance_type):
    if test_utils.is_image_incompatible_with_instance_type(mxnet_training, ec2_instance_type):
        pytest.skip(f"Image {mxnet_training} is incompatible with instance type {ec2_instance_type}")
    execute_ec2_training_test(ec2_connection, mxnet_training, MX_MNIST_CMD)


@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_mxnet_train_mnist_cpu(mxnet_training, ec2_connection, cpu_only):
    execute_ec2_training_test(ec2_connection, mxnet_training, MX_MNIST_CMD)


@pytest.mark.integration("keras")
@pytest.mark.model("resnet")
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_mxnet_keras_gpu(mxnet_training, ec2_connection, gpu_only, ec2_instance_type):
    if test_utils.is_image_incompatible_with_instance_type(mxnet_training, ec2_instance_type):
        pytest.skip(f"Image {mxnet_training} is incompatible with instance type {ec2_instance_type}")
    _, framework_version = test_utils.get_framework_and_version_from_tag(mxnet_training)
    if Version(framework_version) >= Version('1.9.0'):
        pytest.skip(f"Keras support has been deprecated MXNet 1.9.0 onwards")
    execute_ec2_training_test(ec2_connection, mxnet_training, MX_KERAS_CMD)


@pytest.mark.integration("keras")
@pytest.mark.model("resnet")
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_mxnet_keras_cpu(mxnet_training, ec2_connection, cpu_only):
    _, framework_version = test_utils.get_framework_and_version_from_tag(mxnet_training)
    if Version(framework_version) >= Version('1.9.0'):
        pytest.skip(f"Keras support has been deprecated MXNet 1.9.0 onwards")
    execute_ec2_training_test(ec2_connection, mxnet_training, MX_KERAS_CMD)


@pytest.mark.integration("dgl")
@pytest.mark.model("gcn")
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_mxnet_train_dgl_gpu(mxnet_training, ec2_connection, gpu_only, py3_only, ec2_instance_type):
    if test_utils.is_image_incompatible_with_instance_type(mxnet_training, ec2_instance_type):
        pytest.skip(f"Image {mxnet_training} is incompatible with instance type {ec2_instance_type}")
    # TODO: remove/update this when DGL supports MXNet 1.9
    _, framework_version = test_utils.get_framework_and_version_from_tag(mxnet_training)
    if Version(framework_version) >= Version('1.9.0'):
        pytest.skip("Skipping DGL tests as DGL does not yet support MXNet 1.9")
    execute_ec2_training_test(ec2_connection, mxnet_training, MX_DGL_CMD)


@pytest.mark.integration("dgl")
@pytest.mark.model("gcn")
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_mxnet_train_dgl_cpu(mxnet_training, ec2_connection, cpu_only, py3_only):
    execute_ec2_training_test(ec2_connection, mxnet_training, MX_DGL_CMD)


@pytest.mark.integration("gluonnlp")
@pytest.mark.model("textCNN")
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_mxnet_train_nlp_gpu(mxnet_training, ec2_connection, gpu_only, py3_only, ec2_instance_type):
    if test_utils.is_image_incompatible_with_instance_type(mxnet_training, ec2_instance_type):
        pytest.skip(f"Image {mxnet_training} is incompatible with instance type {ec2_instance_type}")
    execute_ec2_training_test(ec2_connection, mxnet_training, MX_NLP_CMD)


@pytest.mark.integration("gluonnlp")
@pytest.mark.model("textCNN")
@pytest.mark.skip(reason="Skip test due to failure on mainline pipeline. See https://github.com/aws/deep-learning-containers/issues/936")
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_mxnet_train_nlp_cpu(mxnet_training, ec2_connection, cpu_only, py3_only):
    execute_ec2_training_test(ec2_connection, mxnet_training, MX_NLP_CMD)


@pytest.mark.integration("horovod")
@pytest.mark.model("AlexNet")
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_mxnet_with_horovod_gpu(mxnet_training, ec2_connection, gpu_only, ec2_instance_type):
    if test_utils.is_image_incompatible_with_instance_type(mxnet_training, ec2_instance_type):
        pytest.skip(f"Image {mxnet_training} is incompatible with instance type {ec2_instance_type}")
    execute_ec2_training_test(ec2_connection, mxnet_training, f"{MX_HVD_CMD} {ec2_instance_type}")


@pytest.mark.integration("horovod")
@pytest.mark.model("AlexNet")
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_mxnet_with_horovod_cpu(mxnet_training, ec2_connection, cpu_only, ec2_instance_type):
    execute_ec2_training_test(ec2_connection, mxnet_training, f"{MX_HVD_CMD} {ec2_instance_type}")


@pytest.mark.flaky(reruns=3)
@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("telemetry")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_SINGLE_GPU_INSTANCE_TYPE, indirect=True)
def test_mxnet_telemetry_gpu(mxnet_training, ec2_connection, gpu_only, ec2_instance_type):
    if test_utils.is_image_incompatible_with_instance_type(mxnet_training, ec2_instance_type):
        pytest.skip(f"Image {mxnet_training} is incompatible with instance type {ec2_instance_type}")
    execute_ec2_training_test(ec2_connection, mxnet_training, MX_TELEMETRY_CMD)


@pytest.mark.flaky(reruns=3)
@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("telemetry")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", MX_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_mxnet_telemetry_cpu(mxnet_training, ec2_connection, cpu_only):
    execute_ec2_training_test(ec2_connection, mxnet_training, MX_TELEMETRY_CMD)
