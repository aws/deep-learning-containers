import os

import test.test_utils as test_utils
import test.test_utils.ec2 as ec2_utils

from test.test_utils import (
    CONTAINER_TESTS_PREFIX,
    is_tf_version,
)
from test.test_utils.ec2 import execute_ec2_training_test, get_ec2_instance_type


TF1_STANDALONE_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testTensorflow1Standalone")
TF2_STANDALONE_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testTensorflow2Standalone")
TF_MNIST_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testTensorFlow")
TF1_HVD_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testTF1HVD")
TF2_HVD_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testTF2HVD")
TF_OPENCV_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testOpenCV")
TF_TELEMETRY_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "test_tf_dlc_telemetry_test")
TF_KERAS_HVD_CMD_AMP = os.path.join(CONTAINER_TESTS_PREFIX, "testTFKerasHVDAMP")
TF_KERAS_HVD_CMD_FP32 = os.path.join(CONTAINER_TESTS_PREFIX, "testTFKerasHVDFP32")
TF_TENSORBOARD_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testTensorBoard")
TF_ADDONS_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testTFAddons")
TF_DATASERVICE_TEST_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testDataservice")
TF_DATASERVICE_DISTRIBUTE_TEST_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX, "testDataserviceDistribute"
)
TF_IO_S3_PLUGIN_TEST_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testTensorflowIoS3Plugin")
TF_HABANA_TEST_SUITE_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testHabanaTFSuite")

TF_EC2_SINGLE_GPU_INSTANCE_TYPE = get_ec2_instance_type(
    default="p3.2xlarge", processor="gpu", filter_function=ec2_utils.filter_only_single_gpu
)
TF_EC2_GPU_INSTANCE_TYPE = get_ec2_instance_type(default="g3.16xlarge", processor="gpu")
TF_EC2_CPU_INSTANCE_TYPE = get_ec2_instance_type(default="c5.9xlarge", processor="cpu")
TF_EC2_HPU_INSTANCE_TYPE = get_ec2_instance_type(default="dl1.24xlarge", processor="hpu")


class TFTrainingTestFailure(Exception):
    pass


def tensorflow_standalone(tensorflow_training, ec2_connection):
    """
    Single GPU sanity test
    """
    test_script = (
        TF1_STANDALONE_CMD if is_tf_version("1", tensorflow_training) else TF2_STANDALONE_CMD
    )
    execute_ec2_training_test(
        ec2_connection, tensorflow_training, test_script, container_name="tf_standalone"
    )


def tensorflow_mnist(tensorflow_training, ec2_connection, gpu_only, ec2_instance_type):
    execute_ec2_training_test(
        ec2_connection, tensorflow_training, TF_MNIST_CMD, container_name="mnist"
    )


def tensorflow_opencv(tensorflow_training, ec2_connection):
    execute_ec2_training_test(
        ec2_connection, tensorflow_training, TF_OPENCV_CMD, container_name="opencv"
    )


def tensorflow_telemetry(tensorflow_training, ec2_connection):
    execute_ec2_training_test(
        ec2_connection, tensorflow_training, TF_TELEMETRY_CMD, container_name="telemetry"
    )


def tensorflow_tensorboard(tensorflow_training, ec2_connection):
    execute_ec2_training_test(
        ec2_connection, tensorflow_training, TF_TENSORBOARD_CMD, container_name="tensorboard"
    )


# TensorFlow Addons is actively working towards forward compatibility with TensorFlow 2.x
# https://github.com/tensorflow/addons#python-op-compatility
def tensorflow_addons(tensorflow_training, ec2_connection):
    execute_ec2_training_test(
        ec2_connection, tensorflow_training, TF_ADDONS_CMD, container_name="addons"
    )


def tensorflow_io_s3_plugin(tensorflow_training, ec2_connection):
    execute_ec2_training_test(
        ec2_connection, tensorflow_training, TF_IO_S3_PLUGIN_TEST_CMD, container_name="s3plugin"
    )


# Helper function to test data service
def run_data_service_test(ec2_connection, tensorflow_training, cmd, container_name="dataservice"):
    _, tensorflow_version = test_utils.get_framework_and_version_from_tag(tensorflow_training)
    ec2_connection.run("python -m pip install --upgrade pip")
    ec2_connection.run(f"python -m pip install tensorflow=={tensorflow_version}")
    ec2_connection.run("python -m pip install 'protobuf<4'")
    container_test_local_dir = os.path.join("$HOME", "container_tests")
    ec2_connection.run(
        f"cd {container_test_local_dir}/bin && screen -d -m python start_dataservice.py"
    )
    execute_ec2_training_test(
        ec2_connection, tensorflow_training, cmd, host_network=True, container_name=container_name
    )


def tensorflow_dataservice(tensorflow_training, ec2_connection):
    run_data_service_test(
        ec2_connection, tensorflow_training, TF_DATASERVICE_TEST_CMD, container_name="dataservice"
    )


def tensorflow_distribute_dataservice(tensorflow_training, ec2_connection):
    run_data_service_test(
        ec2_connection,
        tensorflow_training,
        TF_DATASERVICE_DISTRIBUTE_TEST_CMD,
        container_name="dataservice_dist",
    )
