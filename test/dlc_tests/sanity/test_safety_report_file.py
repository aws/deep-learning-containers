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

SAFETY_FILE = "/opt/aws/dlc/info/safety_report.json"


@dataclass
class SafetyVulnerabilityAdvisory:
    """
    One of the DataClasses for parsing Safety Report
    """

    vulnerability_id: str
    advisory: str
    reason_to_ignore: str
    spec: str


@dataclass
class SafetyPackageVulnerabilityReport:
    """
    One of the DataClasses for parsing Safety Report
    """

    package: str
    scan_status: str
    installed: str
    vulnerabilities: List[SafetyVulnerabilityAdvisory]
    date: str

    def __post_init__(self):
        self.vulnerabilities = [SafetyVulnerabilityAdvisory(**i) for i in self.vulnerabilities]


@dataclass
class SafetyPythonEnvironmentVulnerabilityReport:
    """
    One of the DataClasses for parsing Safety Report
    """

    report: List[SafetyPackageVulnerabilityReport]

    def __post_init__(self):
        self.report = [SafetyPackageVulnerabilityReport(**i) for i in self.report]


@pytest.mark.model("N/A")
@pytest.mark.skipif(is_canary_context(), reason="Skipping test because it does not run on canary")
def test_safety_file_exists_and_is_valid(image):
    """
    Checks if the image has a safety report at the desired location and fails if any of the
    packages in the report have failed the safety check.

    :param image: str, image uri
    """
    repo_name, image_tag = image.split("/")[-1].split(":")
    # Make sure this container name doesn't conflict with the safety check test container name
    container_name = f"{repo_name}-{image_tag}-safety-file"
    # Add null entrypoint to ensure command exits immediately
    run(
        f"docker run -id " f"--name {container_name} " f"--entrypoint='/bin/bash' " f"{image}",
        hide=True,
        warn=True,
    )

    try:
        # Check if file exists
        docker_exec_cmd = f"docker exec -i {container_name}"
        safety_file_check = run(f"{docker_exec_cmd} test -f {SAFETY_FILE}", warn=True, hide=True)
        assert safety_file_check.ok, f"Safety file existence test failed for {image}"

        file_content = run(f"{docker_exec_cmd} cat {SAFETY_FILE}", warn=True, hide=True)
        raw_scan_result = json.loads(file_content.stdout)
        safety_report_object = SafetyPythonEnvironmentVulnerabilityReport(report=raw_scan_result)

        # processing safety reports
        report_log_template = "SAFETY_REPORT ({status}) [pkg: {pkg}] [installed: {installed}] [vulnerabilities: {vulnerabilities}]"
        failed_count = 0
        for report_item in safety_report_object.report:
            if report_item.scan_status == "FAILED":
                failed_count += 1
                LOGGER.error(
                    report_log_template.format(
                        status="FAILED",
                        pkg=report_item.package,
                        installed=report_item.installed,
                        vulnerabilities=report_item.vulnerabilities,
                    )
                )
        assert failed_count == 0, f"{failed_count} package/s failed safety test for {image} !!!"
        LOGGER.info(
            f"Safety report file validation is successfully complete and report exists at {SAFETY_FILE}"
        )
    finally:
        run(f"docker rm -f {container_name}", hide=True, warn=True)


@pytest.mark.model("N/A")
def test_safety_package_not_installed(image):
    """
    We do not ship safety within DLCs. This test confirms that safety is not installed in the image.
    """
    repo_name, image_tag = image.split("/")[-1].split(":")
    # Make sure this container name doesn't conflict with the safety check test container name
    container_name = f"{repo_name}-{image_tag}-safety-package-check"
    # Add null entrypoint to ensure command exits immediately
    try:
        run(
            f"docker run -id " f"--name {container_name} " f"--entrypoint='/bin/bash' " f"{image}",
            hide=True,
        )
        docker_exec_cmd = f"docker exec -i {container_name}"
        safety_package_installed = run(f"{docker_exec_cmd} pip show safety", warn=True, hide=True)
        assert not safety_package_installed.ok, f"Safety is installed in image: {image}"
    finally:
        run(f"docker rm -f {container_name}", hide=True, warn=True)
