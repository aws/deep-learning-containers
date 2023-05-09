from packaging.version import Version
from packaging.specifiers import SpecifierSet
import pytest

from invoke import run
from test.test_utils import get_python_version_from_image_uri


@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.model("N/A")
@pytest.mark.integration("remote_function")
def test_remote_function(image):
    """
    Test to check compatibility of sagemaker training images with sagemaker remote function.
    """
    python_version = get_python_version_from_image_uri(image).replace("py", "")
    python_version = int(python_version)

    if python_version < 37 or "training" not in image:
        pytest.skip(
            f"Skipping remote function compatibility test for {image}. Test only for training images with Python>3.6"
        )

    repo_name, image_tag = image.split("/")[-1].split(":")
    container_name = f"{repo_name}-{image_tag}-remote-function-test"

    run(
        f"docker run -itd --name {container_name} "
        f"--mount type=bind,src=$(pwd)/container_tests,target=/test"
        f" --entrypoint='/bin/bash' "
        f" --env SAGEMAKER_INTERNAL_IMAGE_URI={image}"
        f"{image}",
        echo=True,
        hide=True,
    )
    try:
        docker_exec_cmd = f"docker exec -i {container_name}"
        run(f"{docker_exec_cmd} python /test/bin/test_remote_function.py ", hide=True)
    finally:
        run(f"docker rm -f {container_name}", hide=True)
