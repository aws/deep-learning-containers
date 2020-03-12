"""
ECS tests for PyTorch Training
"""
import pytest
from test.test_utils import ECS_AML2_GPU_USWEST2


def test_ecs_pytorch_training_mnist(request, pytorch_training):
    """
    Placeholder test
    """
    family = request.node.name.split('[')[0]
    print(family)
    print(pytorch_training)
