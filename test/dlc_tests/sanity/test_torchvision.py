import pytest

from invoke.context import Context

from test.test_utils import (
    get_container_name,
    run_cmd_on_container,
    start_container,
)


@pytest.mark.model("N/A")
def test_torchvision_nms_training_cpu(pytorch_training, cpu):
    """
    Check that the internally built torchvision binary is used to resolve the missing nms issue.
    :param pytorch_training: framework fixture for pytorch training
    :param cpu: ECR image URI with "cpu" in the name
    """
    image = cpu
    if "tensorflow" in image or "mxnet" in image:
        pytest.skip(msg="TF and MXNet don't have torchvision installed.")

    ctx = Context()
    container_name = get_container_name("framework-version", image)
    start_container(container_name, image, ctx)
    output = run_cmd_on_container(
        container_name, ctx, f"import torch; import torchvision; print(torch.ops.torchvision.nms)", executable="python"
    )
    assert "RuntimeError" not in output.stdout.strip()
