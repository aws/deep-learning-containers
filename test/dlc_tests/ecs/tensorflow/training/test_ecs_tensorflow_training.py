import os

import pytest

from test.test_utils import ECS_AML2_CPU_USWEST2, ECS_AML2_GPU_USWEST2
from test.test_utils import ecs as ecs_utils
from test.test_utils import ec2 as ec2_utils


TF_MNIST_TRAINING_SCRIPT = os.path.join(os.sep, "test", "bin", "testTensorFlow")


@pytest.mark.parametrize("training_script", [TF_MNIST_TRAINING_SCRIPT], indirect=True)
@pytest.mark.parametrize("ecs_instance_type", ["c5.4xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_CPU_USWEST2], indirect=True)
def test_ecs_tensorflow_training_mnist_cpu(cpu_only, ecs_container_instance, tensorflow_training, training_cmd,
                                           ecs_cluster_name):
    """
    CPU mnist test for TF Training

    Instance Type - c5.4xlarge

    Given above parameters, registers a task with family named after this test, runs the task, and waits for
    the task to be stopped before doing teardown operations of instance and cluster.
    """
    instance_id, cluster_arn = ecs_container_instance

    ecs_utils.ecs_training_test_executor(ecs_cluster_name, cluster_arn, training_cmd, tensorflow_training, instance_id)


@pytest.mark.parametrize("training_script", [TF_MNIST_TRAINING_SCRIPT], indirect=True)
@pytest.mark.parametrize("ecs_instance_type", ["p3.2xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_GPU_USWEST2], indirect=True)
def test_ecs_tensorflow_training_mnist_gpu(gpu_only, ecs_container_instance, tensorflow_training, training_cmd,
                                           ecs_cluster_name):
    """
    GPU mnist test for TF Training

    Instance Type - p3.2xlarge

    Given above parameters, registers a task with family named after this test, runs the task, and waits for
    the task to be stopped before doing teardown operations of instance and cluster.
    """
    instance_id, cluster_arn = ecs_container_instance

    num_gpus = ec2_utils.get_instance_num_gpus(instance_id)

    ecs_utils.ecs_training_test_executor(ecs_cluster_name, cluster_arn, training_cmd, tensorflow_training, instance_id,
                                         num_gpus=num_gpus)
