import json

from enum import IntEnum

from test import test_utils


class CVESeverity(IntEnum):
    UNDEFINED = 0
    INFORMATIONAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


class ECRScanVulnerability:
    def __init__(self, package, vulnerability):
        self.package = package
        self.severity = vulnerability["Severity"]
        self.cve = vulnerability["CVE"]
        self.exceptions = vulnerability["NotApplicableOn"]
        self.reason = vulnerability["Reason"]

    def is_image_excepted(self, image_uri):
        """
        Takes image_uri and returns True if image is included in ignore list for this vulnerability

        :param image_uri:
        :return:
        """
        framework, framework_version = test_utils.get_framework_and_version_from_tag(image_uri)
        image_type = test_utils.get_job_type_from_image(image_uri)
        device_type = test_utils.get_processor_from_image_uri(image_uri)
        python_version = test_utils.get_python_version_from_image_uri(image_uri)
        return self.is_image_config_excepted(framework, framework_version, image_type, device_type, python_version)

    def is_image_config_excepted(self, framework, framework_version, image_type, device_type, python_version):
        """
        Takes image configuration and returns True if image is included in ignore list for this vulnerability

        :param framework:
        :param framework_version:
        :param image_type:
        :param device_type:
        :param python_version:
        :return:
        """
        if (
            framework in self.exceptions.get("framework", [])
            and framework_version in self.exceptions.get("framework_version", [])
            and image_type in self.exceptions.get("image_type", [])
            and python_version in self.exceptions.get("python_version", [])
            and device_type in self.exceptions.get("device_type", [])
        ):
            return True


class ScanAllowList:
    def __init__(self, file_path):
        with open(file_path, "r") as f:
            allowlist_json = json.load(f)
        self.vulnerability_dict = dict()
        for package, vulnerability_list in allowlist_json.items():
            for vulnerability in vulnerability_list:
                if package not in self.vulnerability_dict:
                    self.vulnerability_dict[package] = list()
                self.vulnerability_dict[package].append(ECRScanVulnerability(package, vulnerability))

    def is_package_in_allow_list(self, image_uri, package, cve, severity):
        if package not in self.vulnerability_dict:
            return False
        for vulnerability in self.vulnerability_dict[package]:
            if (
                vulnerability.cve == cve
                and vulnerability.is_image_excepted(image_uri)
                and CVESeverity[vulnerability.severity] == CVESeverity[severity]
            ):
                return True, vulnerability.reason
        return False, ""
