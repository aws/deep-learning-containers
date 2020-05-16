import os
import pytest

from test.test_utils import CONTAINER_TESTS_PREFIX, is_tf1, is_tf2, is_pr_context, get_dlc_images
from test.test_utils.ec2 import execute_ec2_training_test


TF1_STANDALONE_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testTensorflow1Standalone")
TF2_STANDALONE_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testTensorflow2Standalone")
TF_MNIST_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testTensorFlow")
TF1_HVD_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testTF1HVD")
TF2_HVD_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testTF2HVD")
TF_OPENCV_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testOpenCV")


# if is_pr_context():
#     TF_EC2_GPU_INSTANCE_TYPE = ["p2.xlarge"]
#     TF_EC2_CPU_INSTANCE_TYPE = ["c5.4xlarge"]
# else:
TF_EC2_GPU_INSTANCE_TYPE = ["g3.4xlarge", "p2.8xlarge", "p3.16xlarge"]
TF_EC2_CPU_INSTANCE_TYPE = ["c4.8xlarge", "m4.16xlarge", "t2.2xlarge"]

# As our next release is TF2, adding p3dn testing
# images = get_dlc_images()
# if is_tf2(images):
#     TF_EC2_GPU_INSTANCE_TYPE.append("p3dn.24xlarge")


@pytest.mark.skip("temporary")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_standalone_gpu(tensorflow_training, ec2_connection, gpu_only):
    test_script = TF1_STANDALONE_CMD if is_tf1(tensorflow_training) else TF2_STANDALONE_CMD
    execute_ec2_training_test(ec2_connection, tensorflow_training, test_script)


@pytest.mark.skip("temporary")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_standalone_cpu(tensorflow_training, ec2_connection, cpu_only):
    test_script = TF1_STANDALONE_CMD if is_tf1(tensorflow_training) else TF2_STANDALONE_CMD
    execute_ec2_training_test(ec2_connection, tensorflow_training, test_script)


@pytest.mark.skip("temporary")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_train_mnist_gpu(tensorflow_training, ec2_connection, gpu_only):
    execute_ec2_training_test(ec2_connection, tensorflow_training, TF_MNIST_CMD)


@pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_train_mnist_cpu(tensorflow_training, ec2_connection, cpu_only):
    execute_ec2_training_test(ec2_connection, tensorflow_training, TF_MNIST_CMD)


@pytest.mark.skip("temporary")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_with_horovod_gpu(tensorflow_training, ec2_connection, gpu_only):
    test_script = TF1_HVD_CMD if is_tf1(tensorflow_training) else TF2_HVD_CMD
    execute_ec2_training_test(ec2_connection, tensorflow_training, test_script)


@pytest.mark.skip(reason="Skipping the test temporarily due to timeout issue")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_with_horovod_cpu(tensorflow_training, ec2_connection, cpu_only):
    test_script = TF1_HVD_CMD if is_tf1(tensorflow_training) else TF2_HVD_CMD
    execute_ec2_training_test(ec2_connection, tensorflow_training, test_script)


@pytest.mark.skip("temporary")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_opencv_gpu(tensorflow_training, ec2_connection, gpu_only):
    if is_tf1(tensorflow_training):
        pytest.skip("This test is for TF2 only")
    execute_ec2_training_test(ec2_connection, tensorflow_training, TF_OPENCV_CMD)


@pytest.mark.skip("temporary")
@pytest.mark.parametrize("ec2_instance_type", TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_tensorflow_opencv_cpu(tensorflow_training, ec2_connection, cpu_only):
    if is_tf1(tensorflow_training):
        pytest.skip("This test is for TF2 only")
    execute_ec2_training_test(ec2_connection, tensorflow_training, TF_OPENCV_CMD)
