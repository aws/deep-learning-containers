import datetime

import pytest

from test.test_utils import ECS_AML2_CPU_USWEST2


@pytest.mark.parametrize("ecs_instance_type", ["c5.9xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_CPU_USWEST2], indirect=True)
@pytest.mark.parametrize(
    "ecs_cluster_name",
    [f"pt-train-mnist-cluster-{datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S')}"],
    indirect=True,
)
def test_ecs_pytorch_training_mnist_cpu(request, pytorch_training, ecs_container_instance, ecs_client):
    """
    This is a direct test of our ECS PT training documentation.

    Given above parameters, registers a task with family named after this test, runs the task, and waits for
    the task to be stopped before doing teardown operations of instance and cluster.
    """
    # Will remove when gpu fixture is available
    if "gpu" in pytorch_training:
        return

    _instance_id, cluster = ecs_container_instance

    # Naming the family after the test name, which is in this format
    family = request.node.name.split("[")[0]

    container_definitions = [
        {
            "command": [
                "git clone https://github.com/pytorch/examples.git && python examples/mnist/main.py --no-cuda"
            ],
            "entryPoint": ["sh", "-c"],
            "name": "pytorch-training-container",
            "image": pytorch_training,
            "memory": 4000,
            "cpu": 256,
            "essential": True,
            "portMappings": [{"containerPort": 80, "protocol": "tcp"}],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/pytorch-training-cpu",
                    "awslogs-region": "us-west-2",
                    "awslogs-stream-prefix": "mnist",
                    "awslogs-create-group": "true",
                },
            },
        }
    ]

    ecs_client.register_task_definition(
        requiresCompatibilities=["EC2"],
        containerDefinitions=container_definitions,
        volumes=[],
        networkMode="bridge",
        placementConstraints=[],
        family=family,
    )

    task = ecs_client.run_task(cluster=cluster, taskDefinition=family)
    task_arn = task.get("tasks", [{}])[0].get("taskArn")
    waiter = ecs_client.get_waiter("tasks_stopped")
    waiter.wait(cluster=cluster, tasks=[task_arn])
