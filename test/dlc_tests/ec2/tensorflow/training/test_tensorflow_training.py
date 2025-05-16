import re
import os
import pytest

import test.test_utils as test_utils
import test.test_utils.ec2 as ec2_utils

from test.test_utils import (
    CONTAINER_TESTS_PREFIX,
    UBUNTU_18_HPU_DLAMI_US_WEST_2,
    LOGGER,
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
    default="g5.8xlarge", processor="gpu", filter_function=ec2_utils.filter_only_single_gpu
)
TF_EC2_GPU_INSTANCE_TYPE = get_ec2_instance_type(default="g4dn.8xlarge", processor="gpu")
TF_EC2_CPU_INSTANCE_TYPE = get_ec2_instance_type(
    default="c5.9xlarge", processor="cpu", filter_function=ec2_utils.filter_no_t32x
)
TF_EC2_HPU_INSTANCE_TYPE = get_ec2_instance_type(default="dl1.24xlarge", processor="hpu")


class TFTrainingTestFailure(Exception):
    pass


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.skipif(
    not test_utils.is_deep_canary_context() or not os.getenv("REGION") == "us-west-2",
    reason="This test only needs to run in deep-canary context in us-west-2",
)
@pytest.mark.deep_canary("Reason: This test is a simple tf mnist test")
@pytest.mark.model("mnist")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_train_mnist_gpu_deep_canary(
    tensorflow_training, ec2_connection, gpu_only, ec2_instance_type
):
    if test_utils.is_image_incompatible_with_instance_type(tensorflow_training, ec2_instance_type):
        pytest.skip(
            f"Image {tensorflow_training} is incompatible with instance type {ec2_instance_type}"
        )
    execute_ec2_training_test(ec2_connection, tensorflow_training, TF_MNIST_CMD)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.skipif(
    not test_utils.is_deep_canary_context() or not os.getenv("REGION") == "us-west-2",
    reason="This test only needs to run in deep-canary context in us-west-2",
)
@pytest.mark.deep_canary("Reason: This test is a simple tf mnist test")
@pytest.mark.model("mnist")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_train_mnist_cpu_deep_canary(tensorflow_training, ec2_connection, cpu_only):
    execute_ec2_training_test(ec2_connection, tensorflow_training, TF_MNIST_CMD)


@pytest.mark.integration("tensorflow_sanity_test")
@pytest.mark.model("N/A")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_SINGLE_GPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_standalone_gpu(
    tensorflow_training, ec2_connection, gpu_only, ec2_instance_type
):
    if test_utils.is_image_incompatible_with_instance_type(tensorflow_training, ec2_instance_type):
        pytest.skip(
            f"Image {tensorflow_training} is incompatible with instance type {ec2_instance_type}"
        )
    test_script = (
        TF1_STANDALONE_CMD if is_tf_version("1", tensorflow_training) else TF2_STANDALONE_CMD
    )
    execute_ec2_training_test(ec2_connection, tensorflow_training, test_script)


@pytest.mark.integration("tensorflow_sanity_test")
@pytest.mark.model("N/A")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_standalone_cpu(tensorflow_training, ec2_connection, cpu_only):
    test_script = (
        TF1_STANDALONE_CMD if is_tf_version("1", tensorflow_training) else TF2_STANDALONE_CMD
    )
    execute_ec2_training_test(ec2_connection, tensorflow_training, test_script)


@pytest.mark.model("mnist")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_train_mnist_gpu(
    tensorflow_training, ec2_connection, gpu_only, ec2_instance_type
):
    if test_utils.is_image_incompatible_with_instance_type(tensorflow_training, ec2_instance_type):
        pytest.skip(
            f"Image {tensorflow_training} is incompatible with instance type {ec2_instance_type}"
        )
    execute_ec2_training_test(ec2_connection, tensorflow_training, TF_MNIST_CMD)


@pytest.mark.model("mnist")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_train_mnist_cpu(tensorflow_training, ec2_connection, cpu_only):
    execute_ec2_training_test(ec2_connection, tensorflow_training, TF_MNIST_CMD)


@pytest.mark.integration("horovod")
@pytest.mark.model("resnet")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_with_horovod_gpu(
    tensorflow_training, ec2_instance_type, ec2_connection, gpu_only, tf2_only, below_tf213_only
):
    if test_utils.is_image_incompatible_with_instance_type(tensorflow_training, ec2_instance_type):
        pytest.skip(
            f"Image {tensorflow_training} is incompatible with instance type {ec2_instance_type}"
        )
    test_script = TF1_HVD_CMD if is_tf_version("1", tensorflow_training) else TF2_HVD_CMD
    execute_ec2_training_test(
        connection=ec2_connection,
        ecr_uri=tensorflow_training,
        test_cmd=f"{test_script} {ec2_instance_type}",
        large_shm=bool(re.match(r"(g4dn\.12xlarge)", ec2_instance_type)),
    )


@pytest.mark.integration("horovod")
@pytest.mark.model("resnet")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_with_horovod_cpu(
    tensorflow_training, ec2_connection, cpu_only, tf2_only, ec2_instance_type, below_tf213_only
):
    container_name = "tf_hvd_cpu_test"
    test_script = TF1_HVD_CMD if is_tf_version("1", tensorflow_training) else TF2_HVD_CMD
    try:
        execute_ec2_training_test(
            ec2_connection,
            tensorflow_training,
            f"{test_script} {ec2_instance_type}",
            container_name=container_name,
            timeout=1800,
        )
    except Exception as e:
        debug_output = ec2_connection.run(f"docker logs {container_name}")
        debug_stdout = debug_output.stdout
        if "TF HVD tests passed!" in debug_stdout:
            LOGGER.warning(
                f"TF HVD tests succeeded, but there is an issue with fabric. Error:\n{e}\nTest output:\n{debug_stdout}"
            )
            return
        raise TFTrainingTestFailure(f"TF HVD test failed. Full output:\n{debug_stdout}") from e


@pytest.mark.integration("opencv")
@pytest.mark.model("unknown_model")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_opencv_gpu(
    tensorflow_training, ec2_connection, tf2_only, gpu_only, ec2_instance_type
):
    if test_utils.is_image_incompatible_with_instance_type(tensorflow_training, ec2_instance_type):
        pytest.skip(
            f"Image {tensorflow_training} is incompatible with instance type {ec2_instance_type}"
        )
    execute_ec2_training_test(ec2_connection, tensorflow_training, TF_OPENCV_CMD)


@pytest.mark.integration("opencv")
@pytest.mark.model("unknown_model")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_opencv_cpu(tensorflow_training, ec2_connection, tf2_only, cpu_only):
    execute_ec2_training_test(ec2_connection, tensorflow_training, TF_OPENCV_CMD)


# Testing Telemetry Script on only one GPU instance
@pytest.mark.usefixtures("sagemaker")
@pytest.mark.flaky(reruns=3)
@pytest.mark.integration("telemetry")
@pytest.mark.model("N/A")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_SINGLE_GPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_telemetry_gpu(tensorflow_training, ec2_connection, gpu_only, ec2_instance_type):
    if test_utils.is_image_incompatible_with_instance_type(tensorflow_training, ec2_instance_type):
        pytest.skip(
            f"Image {tensorflow_training} is incompatible with instance type {ec2_instance_type}"
        )
    execute_ec2_training_test(ec2_connection, tensorflow_training, TF_TELEMETRY_CMD)


# Testing Telemetry Script on only one CPU instance
@pytest.mark.usefixtures("sagemaker")
@pytest.mark.flaky(reruns=3)
@pytest.mark.integration("telemetry")
@pytest.mark.model("N/A")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_telemetry_cpu(tensorflow_training, ec2_connection, cpu_only):
    execute_ec2_training_test(ec2_connection, tensorflow_training, TF_TELEMETRY_CMD)


# Skip test for TF 2.0 and below: https://github.com/tensorflow/tensorflow/issues/33484#issuecomment-555299647
@pytest.mark.integration("keras, horovod, automatic_mixed_precision (AMP)")
@pytest.mark.model("mnist")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_keras_horovod_amp(
    tensorflow_training,
    ec2_connection,
    tf21_and_above_only,
    gpu_only,
    ec2_instance_type,
    below_tf213_only,
):
    if test_utils.is_image_incompatible_with_instance_type(tensorflow_training, ec2_instance_type):
        pytest.skip(
            f"Image {tensorflow_training} is incompatible with instance type {ec2_instance_type}"
        )
    execute_ec2_training_test(ec2_connection, tensorflow_training, TF_KERAS_HVD_CMD_AMP)


@pytest.mark.integration("keras, horovod, single_precision_floating_point (FP32)")
@pytest.mark.model("mnist")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_keras_horovod_fp32(
    tensorflow_training, ec2_connection, tf2_only, gpu_only, ec2_instance_type, below_tf213_only
):
    if test_utils.is_image_incompatible_with_instance_type(tensorflow_training, ec2_instance_type):
        pytest.skip(
            f"Image {tensorflow_training} is incompatible with instance type {ec2_instance_type}"
        )
    execute_ec2_training_test(ec2_connection, tensorflow_training, TF_KERAS_HVD_CMD_FP32)


# Testing Tensorboard with profiling
@pytest.mark.integration("tensorboard, keras")
@pytest.mark.model("sequential")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_tensorboard_gpu(
    tensorflow_training, ec2_connection, tf2_only, gpu_only, ec2_instance_type
):
    if test_utils.is_image_incompatible_with_instance_type(tensorflow_training, ec2_instance_type):
        pytest.skip(
            f"Image {tensorflow_training} is incompatible with instance type {ec2_instance_type}"
        )
    execute_ec2_training_test(ec2_connection, tensorflow_training, TF_TENSORBOARD_CMD)


# Testing Tensorboard with profiling
@pytest.mark.integration("tensorboard, keras")
@pytest.mark.model("sequential")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_tensorboard_cpu(tensorflow_training, ec2_connection, tf2_only, cpu_only):
    execute_ec2_training_test(ec2_connection, tensorflow_training, TF_TENSORBOARD_CMD)


# TensorFlow Addons is actively working towards forward compatibility with TensorFlow 2.x
# https://github.com/tensorflow/addons#python-op-compatility
# TF-Addons is deprecated and does not work with latest Keras 3. Skipping test for TF2.16 and later versions
@pytest.mark.model("sequential")
@pytest.mark.integration("tensorflow_addons, keras")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_SINGLE_GPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_addons_gpu(
    tensorflow_training, ec2_connection, tf2_only, gpu_only, ec2_instance_type, below_tf216_only
):
    if test_utils.is_image_incompatible_with_instance_type(tensorflow_training, ec2_instance_type):
        pytest.skip(
            f"Image {tensorflow_training} is incompatible with instance type {ec2_instance_type}"
        )
    execute_ec2_training_test(ec2_connection, tensorflow_training, TF_ADDONS_CMD)


# TF-Addons is deprecated and does not work with latest Keras 3. Skipping test for TF2.16 and later versions
@pytest.mark.model("sequential")
@pytest.mark.integration("tensorflow_addons, keras")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_addons_cpu(
    tensorflow_training, ec2_connection, tf2_only, cpu_only, below_tf216_only
):
    execute_ec2_training_test(ec2_connection, tensorflow_training, TF_ADDONS_CMD)


## Skip test for TF2.16/TF2.18/2.19 image due to TF-IO s3 filesystem issue: https://github.com/tensorflow/io/issues/2039
@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("mnist")
@pytest.mark.integration("tensorflow_io, tensorflow_datasets")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_SINGLE_GPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_io_s3_plugin_gpu(
    tensorflow_training,
    ec2_connection,
    tf2_only,
    gpu_only,
    ec2_instance_type,
    skip_tf216,
    skip_tf218,
    skip_tf219,
):
    if test_utils.is_image_incompatible_with_instance_type(tensorflow_training, ec2_instance_type):
        pytest.skip(
            f"Image {tensorflow_training} is incompatible with instance type {ec2_instance_type}"
        )
    execute_ec2_training_test(ec2_connection, tensorflow_training, TF_IO_S3_PLUGIN_TEST_CMD)


## Skip test for TF2.16/TF2.18/TF2.19 image due to TF-IO s3 filesystem issue: https://github.com/tensorflow/io/issues/2039
@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("mnist")
@pytest.mark.integration("tensorflow_io, tensorflow_datasets")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_io_s3_plugin_cpu(
    tensorflow_training,
    ec2_connection,
    tf2_only,
    cpu_only,
    skip_tf216,
    skip_tf218,
    skip_tf219,
):
    execute_ec2_training_test(ec2_connection, tensorflow_training, TF_IO_S3_PLUGIN_TEST_CMD)


# Helper function to test data service
def run_data_service_test(ec2_connection, tensorflow_training, cmd):
    _, tensorflow_version = test_utils.get_framework_and_version_from_tag(tensorflow_training)
    ec2_connection.run(f"python -m pip install --upgrade pip")
    ec2_connection.run(f"python -m pip install tensorflow=={tensorflow_version}")
    ec2_connection.run(f"python -m pip install 'protobuf<4'")
    container_test_local_dir = os.path.join("$HOME", "container_tests")
    ec2_connection.run(
        f"cd {container_test_local_dir}/bin && screen -d -m python start_dataservice.py"
    )
    execute_ec2_training_test(ec2_connection, tensorflow_training, cmd, host_network=True)


# Testing Data Service on only one CPU instance
# Skip test for TF 2.3 and below
@pytest.mark.integration("tensorflow-dataservice-test")
@pytest.mark.model("N/A")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_dataservice_cpu(
    tensorflow_training, ec2_connection, tf24_and_above_only, cpu_only
):
    run_data_service_test(ec2_connection, tensorflow_training, TF_DATASERVICE_TEST_CMD)


# Testing Data Service on only one GPU instance
# Skip test for TF 2.3 and below
@pytest.mark.integration("tensorflow-dataservice-test")
@pytest.mark.model("N/A")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_dataservice_gpu(
    tensorflow_training,
    ec2_connection,
    tf24_and_above_only,
    gpu_only,
    ec2_instance_type,
):
    if test_utils.is_image_incompatible_with_instance_type(tensorflow_training, ec2_instance_type):
        pytest.skip(
            f"Image {tensorflow_training} is incompatible with instance type {ec2_instance_type}"
        )
    run_data_service_test(ec2_connection, tensorflow_training, TF_DATASERVICE_TEST_CMD)


# Testing Data Service Distributed mode on only one CPU instance
# Skip test for TF 2.3 and below
@pytest.mark.integration("tensorflow-dataservice-distribute-test")
@pytest.mark.model("N/A")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_distribute_dataservice_cpu(
    tensorflow_training, ec2_connection, tf24_and_above_only, cpu_only
):
    run_data_service_test(ec2_connection, tensorflow_training, TF_DATASERVICE_DISTRIBUTE_TEST_CMD)


# Testing Data Service Distributed mode on only one GPU instance
# Skip test for TF 2.3 and below
@pytest.mark.integration("tensorflow-dataservice-distribute-test")
@pytest.mark.model("N/A")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_distribute_dataservice_gpu(
    tensorflow_training,
    ec2_connection,
    tf24_and_above_only,
    gpu_only,
    ec2_instance_type,
):
    if test_utils.is_image_incompatible_with_instance_type(tensorflow_training, ec2_instance_type):
        pytest.skip(
            f"Image {tensorflow_training} is incompatible with instance type {ec2_instance_type}"
        )
    run_data_service_test(ec2_connection, tensorflow_training, TF_DATASERVICE_DISTRIBUTE_TEST_CMD)


@pytest.mark.integration("tensorflow-dataservice-distribute-test")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_HPU_INSTANCE_TYPE, indirect=True)
@pytest.mark.parametrize("ec2_instance_ami", [UBUNTU_18_HPU_DLAMI_US_WEST_2], indirect=True)
def test_tensorflow_standalone_hpu(
    tensorflow_training_habana, ec2_connection, upload_habana_test_artifact
):
    execute_ec2_training_test(
        ec2_connection,
        tensorflow_training_habana,
        TF_HABANA_TEST_SUITE_CMD,
        container_name="ec2_training_habana_tensorflow_container",
        enable_habana_async_execution=True,
    )
