import json
import logging
import sys

import pytest

from invoke import run
from dataclasses import dataclass
from typing import List

from test.test_utils import is_canary_context


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stderr))

ECR_FILE = "/opt/aws/dlc/info/ecr_report.json"


@pytest.mark.model("N/A")
@pytest.mark.skipif(is_canary_context(), reason="Skipping test because it does not run on canary")
def test_ecr_file_exists_and_is_valid(image):
    """
    Checks if the image has a ecr report at the desired location and fails if any of the
    packages in the report have failed the inspector check.

    :param image: str, image uri
    """
    repo_name, image_tag = image.split("/")[-1].split(":")
    # Make sure this container name doesn't conflict with the ecr check test container name
    container_name = f"{repo_name}-{image_tag}-ecr-file"
    # Add null entrypoint to ensure command exits immediately
    run(f"docker run -id " f"--name {container_name} " f"--entrypoint='/bin/bash' " f"{image}", hide=True, warn=True)

    try:
        # Check if file exists
        docker_exec_cmd = f"docker exec -i {container_name}"
        ecr_file_check = run(f"{docker_exec_cmd} test -f {ECR_FILE}", warn=True, hide=True)
        assert ecr_file_check.ok, f"ECR file existence test failed for {image}"

        file_content = run(f"{docker_exec_cmd} cat {ECR_FILE}", warn=True, hide=True)
        raw_scan_result = json.loads(file_content.stdout)

        # processing ecr reports
        report_log_template = (
            "ECR_REPORT ({status}) [pkg: {pkg}] [installed: {installed}] [vulnerabilities: {vulnerabilities}]"
        )
        failed_count = 0
        for package in raw_scan_result:
            if package["scan_status"] == "FAILED":
                failed_count += 1
                LOGGER.error(
                    report_log_template.format(
                        status = "FAILED",
                        pkg = package,
                        installed = package["installed"],
                        vulnerabilities = package["vulnerabilities"],
                    )
                )
        assert failed_count == 0, f"{failed_count} package/s failed ecr test for {image} !!!"
        LOGGER.info(f"ECR report file validation is successfully complete and report exists at {ECR_FILE}")
    finally:
        run(f"docker rm -f {container_name}", hide=True, warn=True)
