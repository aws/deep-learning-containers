import pytest

from invoke.context import Context
from packaging.version import Version

from test.test_utils import (
    get_container_name,
    get_framework_and_version_from_tag,
    get_processor_from_image_uri,
    run_cmd_on_container,
    start_container,
)


@pytest.mark.usefixtures("huggingface")
@pytest.mark.model("N/A")
def test_torchvision_nms_training(pytorch_training):
    """
    Check that the internally built torchvision binary is used to resolve the missing nms issue.
    :param pytorch_training: framework fixture for pytorch training
    """
    _, framework_version = get_framework_and_version_from_tag(pytorch_training)
    if Version(framework_version) == Version("1.5.1") and get_processor_from_image_uri(pytorch_training) == "gpu":
        pytest.skip("Skipping this test for PT 1.5.1 GPU Training DLC images")
    ctx = Context()
    container_name = get_container_name("torchvision-nms", pytorch_training)
    start_container(container_name, pytorch_training, ctx)
    run_cmd_on_container(
        container_name, ctx, f"import torch; import torchvision; print(torch.ops.torchvision.nms)", executable="python"
    )


@pytest.mark.model("N/A")
def test_torchvision_nms_inference(pytorch_inference, non_huggingface_only):
    """
    Check that the internally built torchvision binary is used to resolve the missing nms issue.
    :param pytorch_inference: framework fixture for pytorch inference
    """
    _, framework_version = get_framework_and_version_from_tag(pytorch_inference)
    if Version(framework_version) == Version("1.5.1") and get_processor_from_image_uri(pytorch_inference) == "gpu":
        pytest.skip("Skipping this test for PT 1.5.1 GPU Inference DLC images")
    if "eia" in pytorch_inference and Version(framework_version) < Version("1.5.1"):
        pytest.skip("This test does not apply to PT EIA images for PT versions less than 1.5.1")
    if "neuron" in pytorch_inference:
        pytest.skip("Skipping because this is not relevant to PT Neuron images")
    ctx = Context()
    container_name = get_container_name("torchvision-nms", pytorch_inference)
    start_container(container_name, pytorch_inference, ctx)
    run_cmd_on_container(
        container_name, ctx, f"import torch; import torchvision; print(torch.ops.torchvision.nms)", executable="python"
    )
