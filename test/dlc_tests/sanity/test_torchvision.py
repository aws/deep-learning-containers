import pytest

from invoke.context import Context

from test.test_utils import LOGGER


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
    container_name = _get_container_name("framework-version", image)
    _start_container(container_name, image, ctx)
    output = _run_cmd_on_container(
        container_name, ctx, f"import torch; import torchvision; print(torch.ops.torchvision.nms)", executable="python"
    )
    assert "RuntimeError" not in output.stdout.strip()


def _get_container_name(prefix, image_uri):
    """
    Create a unique container name based off of a test related prefix and the image uri
    :param prefix: test related prefix, like "emacs" or "pip-check"
    :param image_uri: ECR image URI
    :return: container name
    """
    return f"{prefix}-{image_uri.split('/')[-1].replace('.', '-').replace(':', '-')}"


def _start_container(container_name, image_uri, context):
    """
    Helper function to start a container locally
    :param container_name: Name of the docker container
    :param image_uri: ECR image URI
    :param context: Invoke context object
    """
    context.run(
        f"docker run --entrypoint='/bin/bash' --name {container_name} -itd {image_uri}", hide=True,
    )


def _run_cmd_on_container(container_name, context, cmd, executable="bash", warn=False):
    """
    Helper function to run commands on a locally running container
    :param container_name: Name of the docker container
    :param context: ECR image URI
    :param cmd: Command to run on the container
    :param executable: Executable to run on the container (bash or python)
    :param warn: Whether to only warn as opposed to exit if command fails
    :return: invoke output, can be used to parse stdout, etc
    """
    if executable not in ("bash", "python"):
        LOGGER.warn(f"Unrecognized executable {executable}. It will be run as {executable} -c '{cmd}'")
    return context.run(
        f"docker exec --user root {container_name} {executable} -c '{cmd}'", hide=True, warn=warn, timeout=60
    )
