import time


def test_dummy(pytorch_training, ecs_container_instance):
    print(pytorch_training)
    print(ecs_container_instance)

    # Sleeping for 300s so I can manually verify the ECS cluster is up with an attached instance
    time.sleep(300)
