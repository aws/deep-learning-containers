import os

import pytest

from test.test_utils import ECS_AML2_CPU_USWEST2, ECS_AML2_GPU_USWEST2, CONTAINER_TESTS_PREFIX
from test.test_utils import ecs as ecs_utils
from test.test_utils import ec2 as ec2_utils


MX_MNIST_TRAINING_SCRIPT = os.path.join(CONTAINER_TESTS_PREFIX, "testMXNet")
MX_DGL_TRAINING_SCRIPT = os.path.join(CONTAINER_TESTS_PREFIX, "dgl_tests", "testMXNetDGL")
MX_GLUONNLP_TRAINING_SCRIPT = os.path.join(CONTAINER_TESTS_PREFIX, "gluonnlp_tests", "testNLP")


@pytest.mark.skip('Not testing this')
@pytest.mark.parametrize("training_script", [MX_MNIST_TRAINING_SCRIPT], indirect=True)
@pytest.mark.parametrize("ecs_instance_type", ["c4.8xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_CPU_USWEST2], indirect=True)
def test_ecs_mxnet_training_mnist_cpu(cpu_only, ecs_container_instance, mxnet_training, training_cmd, ecs_cluster_name):
    """
    CPU mnist test for MXNet Training

    Instance Type - c4.8xlarge

    Given above parameters, registers a task with family named after this test, runs the task, and waits for
    the task to be stopped before doing teardown operations of instance and cluster.
    """
    instance_id, cluster_arn = ecs_container_instance

    ecs_utils.ecs_training_test_executor(ecs_cluster_name, cluster_arn, training_cmd, mxnet_training, instance_id)


@pytest.mark.skip('Not testing this')
@pytest.mark.parametrize("training_script", [MX_MNIST_TRAINING_SCRIPT], indirect=True)
@pytest.mark.parametrize("ecs_instance_type", ["p2.8xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_GPU_USWEST2], indirect=True)
def test_ecs_mxnet_training_mnist_gpu(gpu_only, ecs_container_instance, mxnet_training, training_cmd, ecs_cluster_name):
    """
    GPU mnist test for MXNet Training

    Instance Type - p2.8xlarge

    Given above parameters, registers a task with family named after this test, runs the task, and waits for
    the task to be stopped before doing teardown operations of instance and cluster.
    """
    instance_id, cluster_arn = ecs_container_instance

    num_gpus = ec2_utils.get_instance_num_gpus(instance_id)

    ecs_utils.ecs_training_test_executor(ecs_cluster_name, cluster_arn, training_cmd, mxnet_training, instance_id,
                                         num_gpus=num_gpus)


@pytest.mark.skip('Not testing this')
@pytest.mark.parametrize("training_script", [MX_DGL_TRAINING_SCRIPT], indirect=True)
@pytest.mark.parametrize("ecs_instance_type", ["c4.2xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_CPU_USWEST2], indirect=True)
def test_ecs_mxnet_training_dgl_cpu(cpu_only, py3_only, ecs_container_instance, mxnet_training, training_cmd,
                                    ecs_cluster_name):
    """
    CPU DGL test for MXNet Training

    Instance Type - c4.2xlarge

    DGL is only supported in py3, hence we have used the "py3_only" fixture to ensure py2 images don't run
    on this function.

    Given above parameters, registers a task with family named after this test, runs the task, and waits for
    the task to be stopped before doing teardown operations of instance and cluster.
    """
    instance_id, cluster_arn = ecs_container_instance

    ecs_utils.ecs_training_test_executor(ecs_cluster_name, cluster_arn, training_cmd, mxnet_training, instance_id)


@pytest.mark.skip('Not testing this')
@pytest.mark.parametrize("training_script", [MX_DGL_TRAINING_SCRIPT], indirect=True)
@pytest.mark.parametrize("ecs_instance_type", ["p2.xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_GPU_USWEST2], indirect=True)
def test_ecs_mxnet_training_dgl_gpu(gpu_only, py3_only, ecs_container_instance, mxnet_training, training_cmd,
                                    ecs_cluster_name):
    """
    GPU DGL test for MXNet Training

    Instance Type - p2.xlarge

    DGL is only supported in py3, hence we have used the "py3_only" fixture to ensure py2 images don't run
    on this function.

    Given above parameters, registers a task with family named after this test, runs the task, and waits for
    the task to be stopped before doing teardown operations of instance and cluster.
    """
    instance_id, cluster_arn = ecs_container_instance

    num_gpus = ec2_utils.get_instance_num_gpus(instance_id)

    ecs_utils.ecs_training_test_executor(ecs_cluster_name, cluster_arn, training_cmd, mxnet_training, instance_id,
                                         num_gpus=num_gpus)
