import pytest

from test.test_utils import run_subprocess_cmd
from invoke import run


def test_security(image):
    repo_name, image_tag = image.split('/')[-1].split(':')
    container_name = f"{repo_name}-{image_tag}-security"

    # To avoid cluttering the logs with pull request statements
    run(f"docker pull {image}", hide=True)

    run(f"docker run -itd --name {container_name} --mount " \
                 f"type=bind,src=$(pwd)/container_tests,target=/test" \
                 f" --entrypoint='/bin/bash' " \
                 f"{image}",  echo=True)
    try:
        docker_exec_cmd = f"docker exec -i {container_name}"
        run(f"{docker_exec_cmd} python /test/bin/security_checks.py ")
    finally:
        run(f"docker rm -f {container_name}", hide=True)
        run(f"docker rmi -f {image}", hide=True)


