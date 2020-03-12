"""
ECS tests for PyTorch Training
"""


def test_ecs_pytorch_training_mnist(request, pytorch_training):
    """
    Placeholder test
    """
    family = request.node.name.split('[')[0]
    print(family)
    print(pytorch_training)
