import os

from packaging.version import Version
from packaging.specifiers import SpecifierSet
import pytest

import test.test_utils as test_utils
import test.test_utils.ec2 as ec2_utils

from test.test_utils import CONTAINER_TESTS_PREFIX, get_framework_and_version_from_tag
from test.test_utils.ec2 import execute_ec2_training_test, get_ec2_instance_type


PT_MNIST_INDUCTOR_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchwithInductor")
PT_AMP_INDUCTOR_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchAMPwithInductor")

PT_EC2_GPU_INDUCTOR_INSTANCE_TYPES = []
for instance_type in ["p3.2xlarge", "g5.4xlarge", "g4dn.4xlarge"]:
    PT_EC2_GPU_INDUCTOR_INSTANCE_TYPES.extend(get_ec2_instance_type(
    default=instance_type, processor="gpu", filter_function=ec2_utils.filter_only_single_gpu,))



@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("mnist")
@pytest.mark.integration("inductor")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INDUCTOR_INSTANCE_TYPES, indirect=True)
def test_pytorch_train_mnist_inductor_gpu(pytorch_training, ec2_connection, gpu_only, ec2_instance_type):
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_training)
    if Version(image_framework_version) < Version("2.0"):
        pytest.skip("Torch inductor was introduced in PyTorch 2.0")
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}")
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_MNIST_INDUCTOR_CMD)

@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("nccl")
@pytest.mark.model("resnet18")
@pytest.mark.integration("inductor")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INDUCTOR_INSTANCE_TYPES, indirect=True)
def test_pytorch_nccl(pytorch_training, ec2_connection, gpu_only, py3_only, ec2_instance_type):
    """
    Tests nccl backend
    """
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_training)
    if Version(image_framework_version) < Version("2.0"):
        pytest.skip("Torch inductor was introduced in PyTorch 2.0")
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}")
    test_cmd = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchNccl")
    execute_ec2_training_test(ec2_connection, pytorch_training, test_cmd, large_shm=True)

@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("horovod")
@pytest.mark.integration("inductor")
@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INDUCTOR_INSTANCE_TYPES, indirect=True)
def test_pytorch_with_horovod_inductor(pytorch_training, ec2_connection, gpu_only, ec2_instance_type):
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_training)
    if Version(image_framework_version) < Version("2.0"):
        pytest.skip("Torch inductor was introduced in PyTorch 2.0")
    if 'trcomp' in pytorch_training and Version(image_framework_version) in SpecifierSet("<2.0"):
        pytest.skip(f"Image {pytorch_training} doesn't package horovod. Hence test is skipped.")
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}")
    test_cmd = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPTHVDwithInductor")
    execute_ec2_training_test(ec2_connection, pytorch_training, test_cmd)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("gloo")
@pytest.mark.integration("inductor")
@pytest.mark.model("resnet18")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INDUCTOR_INSTANCE_TYPES, indirect=True)
def test_pytorch_gloo_inductor_gpu(pytorch_training, ec2_connection, gpu_only, py3_only, ec2_instance_type):
    """
    Tests gloo backend with torch inductor
    """
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_training)
    if Version(image_framework_version) < Version("2.0"):
        pytest.skip("Torch inductor was introduced in PyTorch 2.0")
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}")
    test_cmd = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchGlooMpi") + \
        " gloo 1" # backend, inductor flags
    execute_ec2_training_test(ec2_connection, pytorch_training, test_cmd, large_shm=True)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("mpi")
@pytest.mark.integration("inductor")
@pytest.mark.model("resnet18")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INDUCTOR_INSTANCE_TYPES, indirect=True)
def test_pytorch_mpi_inductor_gpu(pytorch_training, ec2_connection, gpu_only, py3_only, ec2_instance_type, pt111_and_above_only, version_skip):
    """
    Tests mpi backend with torch inductor
    """   
    # PT2.0.0 doesn't support MPI https://github.com/pytorch/pytorch/issues/97507
    version_skip(pytorch_training, "2.0.0")
    if Version(image_framework_version) < Version("2.0"):
        pytest.skip("Torch inductor was introduced in PyTorch 2.0")
    if 'trcomp' in pytorch_training:
        pytest.skip(f"Image {pytorch_training} is incompatible with distribution type MPI.")
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}")
    test_cmd = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchGlooMpi") + \
        " mpi 1" # backend, inductor flags
    execute_ec2_training_test(ec2_connection, pytorch_training, test_cmd)

@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("amp")
@pytest.mark.integration("inductor")
@pytest.mark.model("resnet50")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INDUCTOR_INSTANCE_TYPES, indirect=True)
def test_pytorch_amp_inductor(pytorch_training, ec2_connection, gpu_only, ec2_instance_type):
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_training)
    if Version(image_framework_version) < Version("2.0"):
        pytest.skip("Torch inductor was introduced in PyTorch 2.0")
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}")
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_AMP_INDUCTOR_CMD)
