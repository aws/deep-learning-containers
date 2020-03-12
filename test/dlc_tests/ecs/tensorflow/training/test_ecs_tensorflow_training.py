import os
import time

import pytest

from test.test_utils import ECS_AML2_GPU_USWEST2


@pytest.mark.parametrize("ecs_instance_type", ["p2.8xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_GPU_USWEST2], indirect=True)
@pytest.mark.parametrize("ecs_cluster_name", [f"tf-train-mnist-cluster-{os.getenv('TEST_TRIGGER')}"], indirect=True)
def test_ecs_tf_training_mnist(request, tensorflow_training, ecs_container_instance, ecs_client):
    """
    TF training MNIST ECS test

    :param request:
    :param tensorflow_training:
    :param ecs_container_instance:
    :param ecs_client:
    :return:
    """
    _instance_id, cluster = ecs_container_instance
    # Naming the family after the test name, which is in this format
    family = request.node.name.split("[")[0]
    container_definitions = [
        {
            "command": [
                "mkdir -p /test && cd /test && git clone https://github.com/fchollet/keras.git && "
                "chmod +x -R /test/ && python keras/examples/mnist_cnn.py"
            ],
            "entryPoint": ["sh", "-c"],
            "name": "tensorflow-training-container",
            "image": tensorflow_training,
            "memory": 4000,
            "cpu": 256,
            "essential": True,
            "portMappings": [{"containerPort": 80, "protocol": "tcp"}],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "awslogs-tf-ecs",
                    "awslogs-region": "us-east-1",
                    "awslogs-stream-prefix": "tf",
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

    time.sleep(150)

    task = ecs_client.run_task(cluster=cluster, taskDefinition=family)
    task_arn = task.get("tasks", [{}])[0].get("taskArn")
    waiter = ecs_client.get_waiter("tasks_stopped")
    waiter.wait(cluster=cluster, tasks=[task_arn])
