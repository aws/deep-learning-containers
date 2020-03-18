import datetime
import os

import pytest

from test.test_utils import ECS_AML2_CPU_USWEST2, ECS_AML2_GPU_USWEST2
from test.test_utils import ecs as ecs_utils
from test.test_utils import ec2 as ec2_utils


@pytest.mark.parametrize("ecs_instance_type", ["c5.9xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_CPU_USWEST2], indirect=True)
def test_ecs_pytorch_training_mnist_cpu(cpu_only, ecs_container_instance, pytorch_training, s3_artifact_copy,
                                        ecs_cluster_name):
    """
    CPU mnist test for PyTorch Training

    Instance Type - c5.9xlarge

    Given above parameters, registers a task with family named after this test, runs the task, and waits for
    the task to be stopped before doing teardown operations of instance and cluster.
    """
    s3_test_artifact_location = s3_artifact_copy

    training_cmd = ecs_utils.build_ecs_training_command(
        s3_test_artifact_location, os.path.join(os.sep, "test", "bin", "pytorch_tests", "testPyTorch")
    )

    instance_id, cluster = ecs_container_instance

    datestr = datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S')
    num_cpus = ec2_utils.get_instance_num_cpus(instance_id)
    memory = int(ec2_utils.get_instance_memory(instance_id) * 0.8)

    ecs_utils.ecs_training_test_executor(ecs_cluster_name, cluster, datestr, training_cmd, num_cpus, memory,
                                         pytorch_training)


@pytest.mark.parametrize("ecs_instance_type", ["p3.8xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_GPU_USWEST2], indirect=True)
def test_ecs_pytorch_training_mnist_gpu(gpu_only, ecs_container_instance, pytorch_training, s3_artifact_copy,
                                        ecs_cluster_name):
    """
    GPU mnist test for PyTorch Training

    Instance Type - p3.8xlarge

    Given above parameters, registers a task with family named after this test, runs the task, and waits for
    the task to be stopped before doing teardown operations of instance and cluster.
    """
    s3_test_artifact_location = s3_artifact_copy

    training_cmd = ecs_utils.build_ecs_training_command(
        s3_test_artifact_location, os.path.join(os.sep, "test", "bin", "pytorch_tests", "testPyTorch")
    )

    instance_id, cluster = ecs_container_instance

    datestr = datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S')
    num_cpus = ec2_utils.get_instance_num_cpus(instance_id)
    num_gpus = ec2_utils.get_instance_num_gpus(instance_id)
    memory = int(ec2_utils.get_instance_memory(instance_id) * 0.8)

    ecs_utils.ecs_training_test_executor(ecs_cluster_name, cluster, datestr, training_cmd, num_cpus, memory,
                                         pytorch_training, num_gpus=num_gpus)
