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

from safety_check_v2 import safety_check_v2

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

def get_secret():
    """
    Retrieves safety api key from secrets manager
    """
    secret_name = "/codebuild/safety/key"
    secret = ""

    # In this sample we only handle the specific exceptions for the 'GetSecretValue' API.
    # See https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
    # We rethrow the exception by default.

    try:
        secrets_manger_client = boto3.client("secretsmanager", region_name="us-west-2")
        get_secret_value_response = secrets_manger_client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        if e.response["Error"]["Code"] == "DecryptionFailureException":
            # Secrets Manager can't decrypt the protected secret text using the provided KMS key.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response["Error"]["Code"] == "InternalServiceErrorException":
            # An error occurred on the server side.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response["Error"]["Code"] == "InvalidParameterException":
            # You provided an invalid value for a parameter.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response["Error"]["Code"] == "InvalidRequestException":
            # You provided a parameter value that is not valid for the current state of the resource.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response["Error"]["Code"] == "ResourceNotFoundException":
            # We can't find the resource that you asked for.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
    else:
        # Decrypts secret using the associated KMS CMK.
        # Depending on whether the secret is a string or binary, one of these fields will be populated.
        if "SecretString" in get_secret_value_response:
            secret = get_secret_value_response["SecretString"]
        else:
            secret = base64.b64decode(get_secret_value_response["SecretBinary"])

    return secret



parser = argparse.ArgumentParser()
parser.add_argument('--report-path', type=str, help='Safety report path')
parser.add_argument('--ignored-packages', type=str, help='Packages to be ignored')
args = parser.parse_args()

report_path = args.report_path
ignored_packages = args.ignored_packages

# run safety check
scan_results = []

safety_api_key = get_secret()

raw_scan_result = json.loads(
        f"{safety_check_v2} --key {safety_api_key} | jq '.' | tee {report_path}",
)
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

