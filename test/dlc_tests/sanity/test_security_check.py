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

    1. Check when ECR scan was last run
    2. If last scan was less than 24 hours ago, skip scan steps 3, 4, 5.
    3. If last scan was Never or more than 24 hours ago, Start Scan.
    4. For 5 minutes (Run DescribeImages):
    4.1. If imageScanStatus == COMPLETE: exit loop
    4.2. If imageScanStatus == IN_PROGRESS or AttributeNotFound(imageScanStatus): continue loop
    4.3. If imageScanStatus == FAILED: raise RuntimeError
    5. If DescribeImages.imageScanStatus != COMPLETE: raise TimeOutError
    6. If imageScanFindingsSummary.findingSeverityCounts.HIGH/CRITICAL > 0: raise VulnerabilityFoundException
    7. Return Success

    :param image:
    :param ecr_client:
    :return:
    """
    timezone = pytz.timezone("US/Pacific")
    last_scan_time = ecr_utils.get_ecr_image_scan_time(ecr_client, image)
    start_time = datetime.now(tz=timezone)
    scan_status = None
    if not last_scan_time or (start_time - last_scan_time).days >= 1:
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
