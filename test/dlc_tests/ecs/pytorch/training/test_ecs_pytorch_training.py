"""
ECS tests for PyTorch Training
"""


def test_ecs_pytorch_training(request, pytorch_training, ecs_container_instance, ecs_client):
    print(pytorch_training)
    print(ecs_container_instance)

    instance_id, cluster = ecs_container_instance

    family = str(request.node.name)

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
