import os

import pytest

from test.test_utils import ECS_AML2_CPU_USWEST2, ECS_AML2_GPU_USWEST2
from test.test_utils import ecs as ecs_utils
from test.test_utils import ec2 as ec2_utils


@pytest.mark.parametrize("ecs_instance_type", ["c4.8xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_CPU_USWEST2], indirect=True)
def test_ecs_mxnet_training_mnist_cpu(cpu_only, ecs_container_instance, mxnet_training, s3_artifact_copy,
                                      ecs_cluster_name):
    """
    CPU mnist test for MXNet Training

    Instance Type - c4.8xlarge

    Given above parameters, registers a task with family named after this test, runs the task, and waits for
    the task to be stopped before doing teardown operations of instance and cluster.
    """
    s3_test_artifact_location = s3_artifact_copy

    training_cmd = ecs_utils.build_ecs_training_command(
        s3_test_artifact_location, os.path.join(os.sep, "test", "bin", "testMXNet")
    )

    instance_id, cluster_arn = ecs_container_instance

    ecs_utils.ecs_training_test_executor(ecs_cluster_name, cluster_arn, training_cmd, mxnet_training, instance_id)


@pytest.mark.parametrize("ecs_instance_type", ["p2.8xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_GPU_USWEST2], indirect=True)
def test_ecs_mxnet_training_mnist_gpu(gpu_only, ecs_container_instance, mxnet_training, s3_artifact_copy,
                                      ecs_cluster_name):
    """
    GPU mnist test for MXNet Training

    Instance Type - p2.8xlarge

    Given above parameters, registers a task with family named after this test, runs the task, and waits for
    the task to be stopped before doing teardown operations of instance and cluster.
    """
    s3_test_artifact_location = s3_artifact_copy

    training_cmd = ecs_utils.build_ecs_training_command(
        s3_test_artifact_location, os.path.join(os.sep, "test", "bin", "testMXNet")
    )

    instance_id, cluster_arn = ecs_container_instance

    num_gpus = ec2_utils.get_instance_num_gpus(instance_id)

    ecs_utils.ecs_training_test_executor(ecs_cluster_name, cluster_arn, training_cmd, mxnet_training, instance_id,
                                         num_gpus=num_gpus)
