import os
import pytest
from multiprocessing import Process
from fabric import Connection

from test.test_utils import CONTAINER_TESTS_PREFIX, is_tf1, is_tf20
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
TF_DATASERVICE_START_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "startDataservice")
TF_DATASERVICE_TEST_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testDataservice")

TF_EC2_GPU_INSTANCE_TYPE = get_ec2_instance_type(default="p2.8xlarge", processor="gpu")
TF_EC2_CPU_INSTANCE_TYPE = get_ec2_instance_type(default="c5.4xlarge", processor="cpu")


# @pytest.mark.integration('tensorflow_sanity_test')
# @pytest.mark.model("N/A")
# @pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
# def test_tensorflow_standalone_gpu(tensorflow_training, ec2_connection, gpu_only):
#     test_script = TF1_STANDALONE_CMD if is_tf1(tensorflow_training) else TF2_STANDALONE_CMD
#     execute_ec2_training_test(ec2_connection, tensorflow_training, test_script)


# @pytest.mark.integration('tensorflow_sanity_test')
# @pytest.mark.model("N/A")
# @pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
# def test_tensorflow_standalone_cpu(tensorflow_training, ec2_connection, cpu_only):
#     test_script = TF1_STANDALONE_CMD if is_tf1(tensorflow_training) else TF2_STANDALONE_CMD
#     execute_ec2_training_test(ec2_connection, tensorflow_training, test_script)


# @pytest.mark.model("mnist")
# @pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
# def test_tensorflow_train_mnist_gpu(tensorflow_training, ec2_connection, gpu_only):
#     execute_ec2_training_test(ec2_connection, tensorflow_training, TF_MNIST_CMD)


# # TODO: Change this back TF_EC2_CPU_INSTANCE_TYPE. Currently this test times out on c4.8x, m4.16x and t2.2x,
# #       though passes on all three when run manually. For now we are pinning to c5.18 until we can resolve the issue.
# @pytest.mark.model("mnist")
# @pytest.mark.parametrize("ec2_instance_type", ["c5.18xlarge"], indirect=True)
# def test_tensorflow_train_mnist_cpu(tensorflow_training, ec2_connection, cpu_only):
#     execute_ec2_training_test(ec2_connection, tensorflow_training, TF_MNIST_CMD)


# @pytest.mark.integration("horovod")
# @pytest.mark.model("resnet")
# @pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
# def test_tensorflow_with_horovod_gpu(tensorflow_training, ec2_connection, gpu_only):
#     test_script = TF1_HVD_CMD if is_tf1(tensorflow_training) else TF2_HVD_CMD
#     execute_ec2_training_test(ec2_connection, tensorflow_training, test_script,
#                               large_shm=True if "p2.8xlarge" in TF_EC2_GPU_INSTANCE_TYPE else False)


# # TODO: Change this back TF_EC2_CPU_INSTANCE_TYPE. Currently this test times out on c4.8x, m4.16x and t2.2x,
# #       though passes on all three when run manually. For now we are pinning to c5.18 until we can resolve the issue.
# @pytest.mark.integration("horovod")
# @pytest.mark.model("resnet")
# @pytest.mark.parametrize("ec2_instance_type", ["c5.18xlarge"], indirect=True)
# def test_tensorflow_with_horovod_cpu(tensorflow_training, ec2_connection, cpu_only):
#     test_script = TF1_HVD_CMD if is_tf1(tensorflow_training) else TF2_HVD_CMD
#     execute_ec2_training_test(ec2_connection, tensorflow_training, test_script)


# @pytest.mark.integration("opencv")
# @pytest.mark.model("unknown_model")
# @pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
# def test_tensorflow_opencv_gpu(tensorflow_training, ec2_connection, gpu_only):
#     if is_tf1(tensorflow_training):
#         pytest.skip("This test is for TF2 only")
#     execute_ec2_training_test(ec2_connection, tensorflow_training, TF_OPENCV_CMD)


# @pytest.mark.integration("opencv")
# @pytest.mark.model("unknown_model")
# @pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
# def test_tensorflow_opencv_cpu(tensorflow_training, ec2_connection, cpu_only):
#     if is_tf1(tensorflow_training):
#         pytest.skip("This test is for TF2 only")
#     execute_ec2_training_test(ec2_connection, tensorflow_training, TF_OPENCV_CMD)


# # Testing Telemetry Script on only one GPU instance
# @pytest.mark.integration("telemetry")
# @pytest.mark.model("N/A")
# @pytest.mark.parametrize("ec2_instance_type", ["p2.xlarge"], indirect=True)
# def test_tensorflow_telemetry_gpu(tensorflow_training, ec2_connection, gpu_only):
#     execute_ec2_training_test(ec2_connection, tensorflow_training, TF_TELEMETRY_CMD)


# # Testing Telemetry Script on only one CPU instance
# @pytest.mark.integration("telemetry")
# @pytest.mark.model("N/A")
# @pytest.mark.parametrize("ec2_instance_type", ["c5.4xlarge"], indirect=True)
# def test_tensorflow_telemetry_cpu(tensorflow_training, ec2_connection, cpu_only):
#     execute_ec2_training_test(ec2_connection, tensorflow_training, TF_TELEMETRY_CMD)


# @pytest.mark.integration("keras, horovod, automatic_mixed_precision (AMP)")
# @pytest.mark.model("mnist")
# @pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
# def test_tensorflow_keras_horovod_amp(tensorflow_training, ec2_connection, gpu_only):
#     if is_tf1(tensorflow_training) or is_tf20(tensorflow_training):
#         pytest.skip("This test is for TF2.1 and later only") # https://github.com/tensorflow/tensorflow/issues/33484#issuecomment-555299647
#     execute_ec2_training_test(ec2_connection, tensorflow_training, TF_KERAS_HVD_CMD_AMP)


# @pytest.mark.integration("keras, horovod, single_precision_floating_point (FP32)")
# @pytest.mark.model("mnist")
# @pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
# def test_tensorflow_keras_horovod_fp32(tensorflow_training, ec2_connection, gpu_only):
#     if is_tf1(tensorflow_training):
#         pytest.skip("This test is for TF2 and later only")
#     execute_ec2_training_test(ec2_connection, tensorflow_training, TF_KERAS_HVD_CMD_FP32)


# # Testing Tensorboard with profiling
# @pytest.mark.integration("tensorboard, keras")
# @pytest.mark.model("sequential")
# @pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
# def test_tensorflow_tensorboard_gpu(tensorflow_training, ec2_connection, gpu_only):
#     if is_tf1(tensorflow_training):
#         pytest.skip("This test is for TF2 only")
#     execute_ec2_training_test(ec2_connection, tensorflow_training, TF_TENSORBOARD_CMD)


# # Testing Tensorboard with profiling
# @pytest.mark.integration("tensorboard, keras")
# @pytest.mark.model("sequential")
# @pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
# def test_tensorflow_tensorboard_cpu(tensorflow_training, ec2_connection, cpu_only):
#     if is_tf1(tensorflow_training):
#         pytest.skip("This test is for TF2 only")
#     execute_ec2_training_test(ec2_connection, tensorflow_training, TF_TENSORBOARD_CMD)

# # TensorFlow Addons is actively working towards forward compatibility with TensorFlow 2.x
# # https://github.com/tensorflow/addons#python-op-compatility
# @pytest.mark.model("sequential")
# @pytest.mark.integration("tensorflow_addons, keras")
# @pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
# def test_tensorflow_addons_gpu(tensorflow_training, ec2_connection, gpu_only):
#     if is_tf1(tensorflow_training):
#         pytest.skip("This test is for TF2 only")
#     execute_ec2_training_test(ec2_connection, tensorflow_training, TF_ADDONS_CMD)


# @pytest.mark.model("sequential")
# @pytest.mark.integration("tensorflow_addons, keras")
# @pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
# def test_tensorflow_addons_cpu(tensorflow_training, ec2_connection, cpu_only):
#     if is_tf1(tensorflow_training):
#         pytest.skip("This test is for TF2 only")
#     execute_ec2_training_test(ec2_connection, tensorflow_training, TF_ADDONS_CMD)

@pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_dataservice_cpu(tensorflow_training, ec2_connection, cpu_only):
	ec2_connection.run('python3 -m pip install --upgrade pip')
	ec2_connection.run('pip3 install tensorflow')
	ec2_connection.run('pip3 install tf-nightly')
	start_service = Process(target=execute_data_service_start, args=(ec2_connection,))
	start_service.start()

	execute_ec2_training_test(ec2_connection, tensorflow_training, TF_DATASERVICE_TEST_CMD, host_network=True)
	start_service.terminate()

@pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_dataservice_gpu(tensorflow_training, ec2_connection, gpu_only):
	ec2_connection.run('python3 -m pip install --upgrade pip')
	ec2_connection.run('pip3 install tensorflow')
	ec2_connection.run('pip3 install tf-nightly')
	start_service = Process(target=execute_data_service_start, args=(ec2_connection,))
	start_service.start()

	execute_ec2_training_test(ec2_connection, tensorflow_training, TF_DATASERVICE_TEST_CMD, host_network=True)
	start_service.terminate()

def execute_data_service_start(connection):
    container_test_local_dir = os.path.join("$HOME", "container_tests")
    connection.run(f'cd {container_test_local_dir} && python3 bin/start_dataservice.py', timeout=50)
