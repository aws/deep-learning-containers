import pytest

from invoke import run
from test.test_utils import (
    is_pr_context,
    is_functionality_sanity_test_enabled,
)


@pytest.mark.usefixtures("sagemaker", "functionality_sanity")
@pytest.mark.model("N/A")
@pytest.mark.canary("Run security test regularly on production images")
# @pytest.mark.skipif(
#     is_pr_context() and not is_functionality_sanity_test_enabled(),
#     reason="Skip functionality sanity test in PR context if explicitly disabled",
# )
def test_security(image):
    repo_name, image_tag = image.split("/")[-1].split(":")
    container_name = f"{repo_name}-{image_tag}-security"

    run(
        f"docker run -itd --name {container_name} "
        f"--mount type=bind,src=$(pwd)/container_tests,target=/test"
        f" --entrypoint='/bin/bash' "
        f"{image}",
        echo=True,
        hide=True,
    )
    try:
        docker_exec_cmd = f"docker exec -i {container_name}"
        run(f"{docker_exec_cmd} python /test/bin/security_checks.py --image_uri {image}", hide=True)
    finally:
        run(f"docker rm -f {container_name}", hide=True)
