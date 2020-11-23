import os

import pytest

from test.test_utils import ECS_AML2_CPU_USWEST2, ECS_AML2_GPU_USWEST2, CONTAINER_TESTS_PREFIX
from test.test_utils import ecs as ecs_utils
from test.test_utils import ec2 as ec2_utils

from test.test_utils import get_framework_and_version_from_tag, get_cuda_version_from_tag
from packaging.version import Version


PT_MNIST_TRAINING_SCRIPT = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorch")
PT_DGL_TRAINING_SCRIPT = os.path.join(CONTAINER_TESTS_PREFIX, "dgl_tests", "testPyTorchDGL")


@pytest.mark.model("mnist")
@pytest.mark.parametrize("training_script", [PT_MNIST_TRAINING_SCRIPT], indirect=True)
@pytest.mark.parametrize("ecs_instance_type", ["c5.9xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_CPU_USWEST2], indirect=True)
def test_ecs_pytorch_training_mnist_cpu(cpu_only, ecs_container_instance, pytorch_training, training_cmd,
                                        ecs_cluster_name):
    """
    CPU mnist test for PyTorch Training

    Instance Type - c5.9xlarge

    Given above parameters, registers a task with family named after this test, runs the task, and waits for
    the task to be stopped before doing teardown operations of instance and cluster.
    """
    instance_id, cluster_arn = ecs_container_instance

    ecs_utils.ecs_training_test_executor(ecs_cluster_name, cluster_arn, training_cmd, pytorch_training, instance_id)


@pytest.mark.model("mnist")
@pytest.mark.parametrize("training_script", [PT_MNIST_TRAINING_SCRIPT], indirect=True)
@pytest.mark.parametrize("ecs_instance_type", ["p3.8xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_GPU_USWEST2], indirect=True)
def test_ecs_pytorch_training_mnist_gpu(gpu_only, ecs_container_instance, pytorch_training, training_cmd,
                                        ecs_cluster_name):
    """
    GPU mnist test for PyTorch Training

    Instance Type - p3.8xlarge

    Given above parameters, registers a task with family named after this test, runs the task, and waits for
    the task to be stopped before doing teardown operations of instance and cluster.
    """
    instance_id, cluster_arn = ecs_container_instance

    num_gpus = ec2_utils.get_instance_num_gpus(instance_id)

    ecs_utils.ecs_training_test_executor(ecs_cluster_name, cluster_arn, training_cmd, pytorch_training, instance_id,
                                         num_gpus=num_gpus)


@pytest.mark.integration("dgl")
@pytest.mark.model("gcn")
@pytest.mark.parametrize("training_script", [PT_DGL_TRAINING_SCRIPT], indirect=True)
@pytest.mark.parametrize("ecs_instance_type", ["c5.12xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_CPU_USWEST2], indirect=True)
def test_ecs_pytorch_training_dgl_cpu(cpu_only, py3_only, ecs_container_instance, pytorch_training, training_cmd,
                                      ecs_cluster_name):
    """
    CPU DGL test for PyTorch Training

    Instance Type - c5.12xlarge

    DGL is only supported in py3, hence we have used the "py3_only" fixture to ensure py2 images don't run
    on this function.

    Given above parameters, registers a task with family named after this test, runs the task, and waits for
    the task to be stopped before doing teardown operations of instance and cluster.
    """
    instance_id, cluster_arn = ecs_container_instance

    ecs_utils.ecs_training_test_executor(ecs_cluster_name, cluster_arn, training_cmd, pytorch_training, instance_id)


@pytest.mark.integration("dgl")
@pytest.mark.model("gcn")
@pytest.mark.parametrize("training_script", [PT_DGL_TRAINING_SCRIPT], indirect=True)
@pytest.mark.parametrize("ecs_instance_type", ["p3.8xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_GPU_USWEST2], indirect=True)
def test_ecs_pytorch_training_dgl_gpu(gpu_only, py3_only, ecs_container_instance, pytorch_training, training_cmd,
                                      ecs_cluster_name):
    """
    GPU DGL test for PyTorch Training

    Instance Type - p3.8xlarge

    DGL is only supported in py3, hence we have used the "py3_only" fixture to ensure py2 images don't run
    on this function.

    Given above parameters, registers a task with family named after this test, runs the task, and waits for
    the task to be stopped before doing teardown operations of instance and cluster.
    """
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_training)
    image_cuda_version = get_cuda_version_from_tag(pytorch_training)
    if Version(image_framework_version) == Version("1.6") and image_cuda_version == "cu110":
        pytest.skip("DGL does not suport CUDA 11 for PyTorch 1.6")

    instance_id, cluster_arn = ecs_container_instance

    num_gpus = ec2_utils.get_instance_num_gpus(instance_id)

    ecs_utils.ecs_training_test_executor(ecs_cluster_name, cluster_arn, training_cmd, pytorch_training, instance_id,
                                         num_gpus=num_gpus)
