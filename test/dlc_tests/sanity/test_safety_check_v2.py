import json
import logging
import sys

import pytest

from invoke import run
from dataclasses import dataclass
from typing import List

from test.test_utils import (
    CONTAINER_TESTS_PREFIX, is_dlc_cicd_context, is_canary_context, is_mainline_context, is_time_for_canary_safety_scan
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


@dataclass
class SafetyPackageVulnerabilityReport:
    package: str
    affected: str
    installed: str
    vulnerabilities: List[SafetyVulnerabilityAdvisory]

    def __post_init__(self):
        self.vulnerabilities = [SafetyVulnerabilityAdvisory(**i) for i in self.vulnerabilities]


@dataclass
class SafetyPythonEnvironmentVulnerabilityReport:
    report: List[SafetyPackageVulnerabilityReport]

    def __post_init__(self):
        self.report = [SafetyPackageVulnerabilityReport(**i) for i in self.report]



@pytest.mark.model("N/A")
@pytest.mark.canary("Run safety tests regularly on production images")
# @pytest.mark.skipif(not is_dlc_cicd_context(), reason="Skipping test because it is not running in dlc cicd infra")
def test_safety_file_exists(image):
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

        ignored_packages = []

        # processing safety reports
        report_log_template = "SAFETY_REPORT ({status}) [pkg: {pkg}] [installed: {installed}] [affected: {affected}] [reason: {reason}] [env: {env}]"
        failed_count = 0
        for result in scan_results:
            for report_item in result.report:
                if "PASSED_SAFETY_CHECK" not in report_item.affected:
                    if (report_item.package not in ignored_packages) or (
                        report_item.package in ignored_packages
                        and report_item.installed != ignored_packages[report_item.package]
                    ):
                        failed_count += 1
                        print(
                            report_log_template.format(
                                status="FAILED",
                                pkg=report_item.package,
                                installed=report_item.installed,
                                affected=report_item.affected,
                                reason=None,
                                env=result.environment,
                            )
                        )
        assert failed_count == 0, f"Found {failed_count} vulnerabilities. Safety check failed! for {image}"
        LOGGER.info(f"Safety check is complete as a part of docker build and report exist at {SAFETY_FILE}")
    finally:
        run(f"docker rm -f {container_name}", hide=True)
