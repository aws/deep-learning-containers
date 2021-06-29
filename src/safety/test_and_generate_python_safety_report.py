"""
Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import json
from dataclasses import dataclass
from typing import List
import boto3
from botocore.exceptions import ClientError
import base64
import argparse
import subprocess

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

parser = argparse.ArgumentParser()
parser.add_argument('--safety-key', type=str, help='Safety key')
parser.add_argument('--report-path', type=str, help='Safety report path')
parser.add_argument('--ignored-packages', type=str, default=None, help='Packages to be ignored')
args = parser.parse_args()

report_path = args.report_path
ignored_packages = args.ignored_packages
safety_api_key = args.safety_key

# run safety check
scan_results = []

output = subprocess.check_output(f"python3 safety_check_v2.py --key '{safety_api_key}' | jq '.' | tee '{report_path}'", shell=True, executable="/bin/bash")
raw_scan_result = json.loads(output)
scan_results.append(
    SafetyPythonEnvironmentVulnerabilityReport(
        report=raw_scan_result
    )
)

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

if failed_count > 0:
    print(f"ERROR: {failed_count} packages")
    print("Please check the test output and patch vulnerable packages.")
    raise Exception("Safety check failed!")
else:
    print("Passed safety check.")

