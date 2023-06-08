import pytest

from invoke.context import Context
from test import test_utils


@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.model("N/A")
@pytest.mark.integration("remote_function")
def test_remote_function(training):
    """
    Test to check compatibility of sagemaker training images with sagemaker remote function.
    """
    python_version = test_utils.get_python_version_from_image_uri(training).replace("py", "")
    python_version = int(python_version)

    if python_version < 37:
        pytest.skip(
            f"Skipping remote function compatibility test for {training}. Test only for training images with Python>3.6"
        )

    container_name = test_utils.get_container_name("remote-function-test", training)
    ctx = Context()

    ctx.run(
        f"docker run -itd --name {container_name} "
        f"-e AWS_DEFAULT_REGION -e AWS_CONTAINER_CREDENTIALS_RELATIVE_URI "
        f"--mount type=bind,src=$(pwd)/container_tests,target=/test"
        f" --entrypoint='/bin/bash' "
        f"--env SAGEMAKER_INTERNAL_IMAGE_URI={training} "
        f"{training}",
        hide=True,
    )
    try:
        test_utils.run_cmd_on_container(
            container_name, ctx, "python /test/bin/test_remote_function.py", timeout=480
        )
    finally:
        test_utils.stop_and_remove_container(container_name, ctx)
