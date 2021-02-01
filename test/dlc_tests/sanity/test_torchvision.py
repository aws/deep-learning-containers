import pytest

from invoke.context import Context

from test.test_utils import (
    get_container_name,
    run_cmd_on_container,
    start_container,
)


@pytest.mark.model("N/A")
def test_torchvision_nms_training(pytorch_training):
    """
    Check that the internally built torchvision binary is used to resolve the missing nms issue.
    :param pytorch_training: framework fixture for pytorch training
    """
    image = pytorch_training
    ctx = Context()
    container_name = get_container_name("torchvision-nms", image)
    start_container(container_name, image, ctx)
    run_cmd_on_container(
        container_name, ctx, f"import torch; import torchvision; print(torch.ops.torchvision.nms)", executable="python"
    )


@pytest.mark.model("N/A")
def test_torchvision_nms_inference(pytorch_inference):
    """
    Check that the internally built torchvision binary is used to resolve the missing nms issue.
    :param pytorch_inference: framework fixture for pytorch inference
    """
    image = pytorch_inference
    ctx = Context()
    container_name = get_container_name("torchvision-nms", image)
    start_container(container_name, image, ctx)
    run_cmd_on_container(
        container_name, ctx, f"import torch; import torchvision; print(torch.ops.torchvision.nms)", executable="python"
    )
