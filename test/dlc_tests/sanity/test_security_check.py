from datetime import datetime
from time import sleep

import pytest
import pytz

from invoke import run

from test.test_utils import ecr as ecr_utils


@pytest.mark.model("N/A")
@pytest.mark.canary("Run security test regularly on production images")
def test_security(image):
    repo_name, image_tag = image.split("/")[-1].split(":")
    container_name = f"{repo_name}-{image_tag}-security"

    run(
        f"docker run -itd --name {container_name} "
        f"--mount type=bind,src=$(pwd)/container_tests,target=/test"
        f" --entrypoint='/bin/bash' "
        f"{image}",
        echo=True,
    )
    try:
        docker_exec_cmd = f"docker exec -i {container_name}"
        run(f"{docker_exec_cmd} python /test/bin/security_checks.py ")
    finally:
        run(f"docker rm -f {container_name}", hide=True)


@pytest.mark.model("N/A")
def test_ecr_scan(image, ecr_client):
    """
    Run ECR Scan Tool on an image being tested, and raise Error if vulnerabilities found
    1. Start Scan.
    2. For 5 minutes (Run DescribeImages):
    3.1. If imageScanStatus == COMPLETE: exit loop
    3.2. If imageScanStatus == IN_PROGRESS or AttributeNotFound(imageScanStatus): continue loop
    3.3. If imageScanStatus == FAILED: raise RuntimeError
    4. If DescribeImages.imageScanStatus != COMPLETE: raise TimeOutError
    5. assert imageScanFindingsSummary.findingSeverityCounts.HIGH/CRITICAL == 0

    :param image: str Image URI for image to be tested
    :param ecr_client: boto3 Client for ECR
    """
    timezone = pytz.timezone("US/Pacific")
    scan_status = None
    start_time = datetime.now(tz=timezone)
    ecr_utils.start_ecr_image_scan(ecr_client, image)
    while (datetime.now(tz=timezone) - start_time).seconds <= 600:
        scan_status, scan_status_description = ecr_utils.get_ecr_image_scan_status(ecr_client, image)
        if scan_status == "FAILED" or scan_status not in [None, "IN_PROGRESS", "COMPLETE"]:
            raise RuntimeError(scan_status_description)
        if scan_status == "COMPLETE":
            break
        sleep(1)
    if scan_status != "COMPLETE":
        raise TimeoutError(f"ECR Scan is still in {scan_status} state. Exiting.")
    severity_counts = ecr_utils.get_ecr_image_scan_severity_count(ecr_client, image)
    assert not (
        severity_counts.get("HIGH", 0) or severity_counts.get("CRITICAL", 0)
    ), f"Found vulnerabilities in image {image}: {str(severity_counts)}"
