import json

from datetime import datetime
from time import sleep, time

import pytest

from packaging.version import Version

from invoke import run
from invoke import Context

from test.test_utils import get_account_id_from_image_uri, get_region_from_image_uri, login_to_ecr_registry
from test.test_utils import ecr as ecr_utils
from test.test_utils.ecr import CVESeverity


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
@pytest.mark.canary("Run ECR Scan test regularly on production images")
@pytest.mark.integration("check OS dependencies")
def test_ecr_scan(image, ecr_client, sts_client, region):
    """
    Run ECR Scan Tool on an image being tested, and raise Error if vulnerabilities found
    1. Start Scan.
    2. For 5 minutes (Run DescribeImages):
       (We run this for 5 minutes because the Scan is expected to complete in about 2 minutes, though no
        analysis has been performed on exactly how long the Scan takes for a DLC image. Therefore we also
        have a 3 minute buffer beyond the expected amount of time taken.)
    3.1. If imageScanStatus == COMPLETE: exit loop
    3.2. If imageScanStatus == IN_PROGRESS or AttributeNotFound(imageScanStatus): continue loop
    3.3. If imageScanStatus == FAILED: raise RuntimeError
    4. If DescribeImages.imageScanStatus != COMPLETE: raise TimeOutError
    5. assert imageScanFindingsSummary.findingSeverityCounts.HIGH/CRITICAL == 0

    :param image: str Image URI for image to be tested
    :param ecr_client: boto3 Client for ECR
    :param sts_client: boto3 Client for STS
    :param region: str Name of region where test is executed
    """
    test_account_id = sts_client.get_caller_identity().get("Account")
    image_account_id = get_account_id_from_image_uri(image)
    if image_account_id != test_account_id:
        image_repo_uri, image_tag = image.split(":")
        _, image_repo_name = image_repo_uri.split("/")
        target_image_repo_name = f"beta-{image_repo_name}"
        image = ecr_utils.reupload_image_to_test_ecr(image, target_image_repo_name, region)
    minimum_sev_threshold = "MEDIUM"
    scan_status = None
    start_time = time()
    ecr_utils.start_ecr_image_scan(ecr_client, image)
    while (time() - start_time) <= 600:
        scan_status, scan_status_description = ecr_utils.get_ecr_image_scan_status(ecr_client, image)
        if scan_status == "FAILED" or scan_status not in [None, "IN_PROGRESS", "COMPLETE"]:
            raise RuntimeError(scan_status_description)
        if scan_status == "COMPLETE":
            break
        sleep(1)
    if scan_status != "COMPLETE":
        raise TimeoutError(f"ECR Scan is still in {scan_status} state. Exiting.")
    severity_counts = ecr_utils.get_ecr_image_scan_severity_count(ecr_client, image)
    scan_results = ecr_utils.get_ecr_image_scan_results(ecr_client, image, minimum_vulnerability=minimum_sev_threshold)
    assert all(
        count == 0 for sev, count in severity_counts.items() if CVESeverity[sev] >= CVESeverity[minimum_sev_threshold]
    ), (
        f"Found vulnerabilities in image {image}: {str(severity_counts)}\n"
        f"Vulnerabilities: {json.dumps(scan_results, indent=4)}"
    )
