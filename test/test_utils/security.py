import json
import os

from enum import IntEnum

from test import test_utils


class CVESeverity(IntEnum):
    UNDEFINED = 0
    INFORMATIONAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


class ScanAllowList:
    """
    ScanAllowList is a class that reads an OS vulnerability allow-list, in the format stored on the DLC repo,
    to allow easy comparison of any ECR Scan Vulnerabilities on an image with its corresponding allow-list file.
    """
    def __init__(self, file_path):
        """
        Read JSON file and prepare the object with all allowed vulnerabilities

        :param file_path: Path to the allow-list JSON file.
        """
        if not os.path.exists(file_path):
            self.allowlist = {}
            return
        with open(file_path, "r") as f:
            self.allowlist = json.load(f)

    def __contains__(self, vulnerability):
        """
        Check if an input vulnerability exists on the allow-list

        :param vulnerability: dict JSON object consisting of information about the vulnerability in the format
                              presented by the ECR Scan Tool
        :return: bool True if the vulnerability is allowed on the allow-list.
        """
        package = get_ecr_vulnerability_package_name(vulnerability)
        if package not in self.allowlist:
            return False
        for allowed_vulnerability in self.allowlist[package]:
            if (vulnerability["name"], vulnerability["severity"]) == (
                allowed_vulnerability["name"], allowed_vulnerability["severity"]
            ):
                # Do not compare package_version, because this is obtained at the time the CVE was first observed
                # on the ECR Scan.
                if all(
                    attribute in allowed_vulnerability["attributes"]
                    for attribute in vulnerability["attributes"] if not attribute["key"] == "package_version"
                ):
                    return True
        return False


def get_ecr_vulnerability_package_name(vulnerability):
    """
    Get Package Name from a vulnerability JSON object

    :param vulnerability: dict JSON object consisting of information about the vulnerability in the format
                          presented by the ECR Scan Tool
    :return: str package name
    """
    for attribute in vulnerability["attributes"]:
        if attribute["key"] == "package_name":
            return attribute["value"]
    return None


def get_ecr_vulnerability_package_version(vulnerability):
    """
    Get Package Version from a vulnerability JSON object

    :param vulnerability: dict JSON object consisting of information about the vulnerability in the format
                          presented by the ECR Scan Tool
    :return: str package version
    """
    for attribute in vulnerability["attributes"]:
        if attribute["key"] == "package_version":
            return attribute["value"]
    return None
