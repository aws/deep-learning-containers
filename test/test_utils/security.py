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
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            self.allowlist = {}
            return
        with open(file_path, "r") as f:
            self.allowlist = json.load(f)

    def __contains__(self, vulnerability):
        package = get_ecr_vulnerability_package_name(vulnerability)
        if package not in self.allowlist:
            return False
        for allowed_vulnerability in self.allowlist[package]:
            if (vulnerability["name"], vulnerability["severity"]) == (
                allowed_vulnerability["name"], allowed_vulnerability["severity"]
            ):
                if test_utils.are_json_objects_equivalent(
                    vulnerability["attributes"], allowed_vulnerability["attributes"]
                ):
                    return True
        return False


def get_ecr_vulnerability_package_name(vulnerability):
    for attribute in vulnerability["attributes"]:
        if attribute["key"] == "package_name":
            return attribute["value"]
    return None


def get_ecr_vulnerability_package_version(vulnerability):
    for attribute in vulnerability["attributes"]:
        if attribute["key"] == "package_version":
            return attribute["value"]
    return None
