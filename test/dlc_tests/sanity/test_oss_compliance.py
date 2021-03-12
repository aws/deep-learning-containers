import json
import logging
import os
import sys

import pytest
import requests

from invoke import run
from test.test_utils import LOGGER, is_mainline_context
from test.test_utils import CONTAINER_TESTS_PREFIX

from invoke.context import Context

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stderr))

@pytest.mark.integration("oss_compliance")
@pytest.mark.model("N/A")
def test_oss_compliance(image):
    """
    Runs oss_compliance check on a container to check if license attribution file exists
    """

    container_name = f"{repo_name}-{image_tag}-oss_compliance"
    # docker_exec_cmd = f"docker exec -i {container_name}"
    ctx = Context()
    test_file_path = os.path.join(CONTAINER_TESTS_PREFIX, "testOSSCompliance")
    file = "THIRD_PARTY_SOURCE_CODE_URLS"

    # Add null entrypoint to ensure command exits immediately
    run(f"docker run -id "
        f"--name {container_name} "
        f"--mount type=bind,src=$(pwd)/container_tests,target=/test "
        f"--entrypoint='/bin/bash' "
        f"{image}", hide=True)
    try:
        run(f"docker cp {container_name}:/root/{file} ~/{file}")
    finally:
        run(f"docker rm -f {container_name}", hide=True)


    context.run(
        f"docker exec --user root {container_name} bash -c '{test_file_path}'", hide=True, timeout=60
    )
