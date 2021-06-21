import argparse
import json

from datetime import datetime
from time import sleep, time

import boto3
import pytest

from packaging.version import Version

from invoke import run
from invoke import Context

from test.test_utils import get_account_id_from_image_uri, get_region_from_image_uri, login_to_ecr_registry, LOGGER
from test.test_utils import ecr as ecr_utils
from test.test_utils.security import CVESeverity, ECRScanVulnerability, ScanAllowList


def parse_args():
    parser = argparse.ArgumentParser(description="Specify image to scan")
    parser.add_argument("--image", type=str, required=True)
    return parser.parse_args()


def check_image_vulnerabilities(image):
    ecr_client = boto3.client("ecr", region_name="us-west-2")
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
    LOGGER.info(json.dumps(scan_results, indent=4))

    # for vulnerability in scan_results:



if __name__ == "__main__":
    args = parse_args()
    check_image_vulnerabilities(args.image)
