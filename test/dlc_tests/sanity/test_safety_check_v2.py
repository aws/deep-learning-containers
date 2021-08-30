import json
import logging
import sys

import pytest

from invoke import run
from dataclasses import dataclass
from typing import List

from test.test_utils import (
     is_canary_context
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stderr))

SAFETY_FILE = '/var/safety_report.json'
SAFETY_FILE_EXISTS = 0


@dataclass
class SafetyVulnerabilityAdvisory:
    vid: str
    advisory: str
    reason_to_ignore: str
    spec: str


@dataclass
class SafetyPackageVulnerabilityReport:
    package: str
    scan_status: str
    installed: str
    vulnerabilities: List[SafetyVulnerabilityAdvisory]
    date: str

    def __post_init__(self):
        self.vulnerabilities = [SafetyVulnerabilityAdvisory(**i) for i in self.vulnerabilities]


@dataclass
class SafetyPythonEnvironmentVulnerabilityReport:
    report: List[SafetyPackageVulnerabilityReport]

    def __post_init__(self):
        self.report = [SafetyPackageVulnerabilityReport(**i) for i in self.report]



@pytest.mark.model("N/A")
@pytest.mark.skipif(
    is_canary_context(), 
    reason="Skipping test because it is not required to run it on canaries. test_safety_check.py runs on canaries."
)
def test_safety_file_exists_and_is_valid(image):
    repo_name, image_tag = image.split('/')[-1].split(':')
    container_name = f"{repo_name}-{image_tag}-safety"
    # Add null entrypoint to ensure command exits immediately
    run(f"docker run -id "
        f"--name {container_name} "
        f"--entrypoint='/bin/bash' "
        f"{image}", hide=True)
    
    try:
        # Check if file exists
        docker_exec_cmd = f"docker exec -i {container_name}"
        safety_file_check = run(f"{docker_exec_cmd} test -f {SAFETY_FILE}", warn=True, hide=True)
        assert safety_file_check.return_code == SAFETY_FILE_EXISTS,  f"Safety file existence test failed for {image}"

        file_content = run(f"{docker_exec_cmd} cat {SAFETY_FILE}", warn=True, hide=True)
        raw_scan_result = json.loads(file_content.stdout)
        scan_results = []
        scan_results.append(
            SafetyPythonEnvironmentVulnerabilityReport(
                report=raw_scan_result
            )
        )

        # processing safety reports
        report_log_template = "SAFETY_REPORT ({status}) [pkg: {pkg}] [installed: {installed}] [vulnerabilities: {vulnerabilities}]"
        failed_count = 0
        for result in scan_results:
            for report_item in result.report:
                if report_item.scan_status == "FAILED": 
                        failed_count += 1
                        LOGGER.info(
                            report_log_template.format(
                                status="FAILED",
                                pkg=report_item.package,
                                installed=report_item.installed,
                                vulnerabilities = report_item.vulnerabilities,
                            )
                        )
        assert failed_count == 0, f"{failed_count} package/s failed safety test for {image} !!!"
        LOGGER.info(f"Safety check is successfully complete and report exists at {SAFETY_FILE}")
    finally:
        run(f"docker rm -f {container_name}", hide=True)
