import json

from enum import IntEnum

from test import test_utils


class ECRScanFailureException(Exception):
    """
    Base class for other exceptions
    """

    pass


class CVESeverity(IntEnum):
    UNDEFINED = 0
    INFORMATIONAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


class ScanVulnerabilityList:
    """
    ScanAllowList is a class that reads an OS vulnerability allow-list, in the format stored on the DLC repo,
    to allow easy comparison of any ECR Scan Vulnerabilities on an image with its corresponding allow-list file.
    """

    def __init__(self, minimum_severity=CVESeverity["MEDIUM"]):
        self.vulnerability_list = {}
        self.minimum_severity = minimum_severity

    def construct_allowlist_from_file(self, file_path):
        """
        Read JSON file and prepare the object with all allowed vulnerabilities

        :param file_path: Path to the allow-list JSON file.
        :return: dict self.vulnerability_list
        """
        with open(file_path, "r") as f:
            file_allowlist = json.load(f)
        for package_name, package_vulnerability_list in file_allowlist.items():
            for vulnerability in package_vulnerability_list:
                if CVESeverity[vulnerability["severity"]] >= self.minimum_severity:
                    if package_name not in self.vulnerability_list:
                        self.vulnerability_list[package_name] = []
                    self.vulnerability_list[package_name].append(vulnerability)
        return self.vulnerability_list

    def construct_allowlist_from_ecr_scan_result(self, vulnerability_list):
        """
        Read a vulnerability list and construct the vulnerability_list

        :param vulnerability_list: list ECR Scan Result results
        :return: dict self.vulnerability_list
        """
        for vulnerability in vulnerability_list:
            package_name = get_ecr_vulnerability_package_name(vulnerability)
            if package_name not in self.vulnerability_list:
                self.vulnerability_list[package_name] = []
            if CVESeverity[vulnerability["severity"]] >= self.minimum_severity:
                self.vulnerability_list[package_name].append(vulnerability)
        return self.vulnerability_list

    def __contains__(self, vulnerability):
        """
        Check if an input vulnerability exists on the allow-list

        :param vulnerability: dict JSON object consisting of information about the vulnerability in the format
                              presented by the ECR Scan Tool
        :return: bool True if the vulnerability is allowed on the allow-list.
        """
        package_name = get_ecr_vulnerability_package_name(vulnerability)
        if package_name not in self.vulnerability_list:
            return False
        for allowed_vulnerability in self.vulnerability_list[package_name]:
            if are_vulnerabilities_equivalent(vulnerability, allowed_vulnerability):
                return True
        return False

    def __cmp__(self, other):
        """
        Compare two ScanVulnerabilityList objects for equivalence

        :param other: Another ScanVulnerabilityList object
        :return: True if equivalent, False otherwise
        """
        if not other or not other.vulnerability_list:
            return not self.vulnerability_list

        if sorted(self.vulnerability_list.keys()) != sorted(other.vulnerability_list.keys()):
            return False

        for package_name, package_vulnerabilities in self.vulnerability_list.items():
            if len(self.vulnerability_list[package_name]) != len(other.vulnerability_list[package_name]):
                return False
            for v1, v2 in zip(
                sorted(self.vulnerability_list[package_name]), sorted(other.vulnerability_list[package_name])
            ):
                if not are_vulnerabilities_equivalent(v1, v2):
                    return False
        return True

    def __sub__(self, other):
        """
        Difference between ScanVulnerabilityList objects

        :param other: Another ScanVulnerabilityList object
        :return: List of vulnerabilities that exist in self, but not in other
        """
        if not self.vulnerability_list:
            return None
        if not other or not other.vulnerability_list:
            return self
        missing_vulnerabilities = [
            vulnerability
            for package_vulnerabilities in self.vulnerability_list.values()
            for vulnerability in package_vulnerabilities
            if vulnerability not in other
        ]
        if not missing_vulnerabilities:
            return None

        difference = ScanVulnerabilityList(minimum_severity=self.minimum_severity)
        difference.construct_allowlist_from_ecr_scan_result(missing_vulnerabilities)
        return difference


def are_vulnerabilities_equivalent(vulnerability_1, vulnerability_2):
    """
    Check if two vulnerability JSON objects are equivalent

    :param vulnerability_1: dict JSON object consisting of information about the vulnerability in the format
                            presented by the ECR Scan Tool
    :param vulnerability_2: dict JSON object consisting of information about the vulnerability in the format
                            presented by the ECR Scan Tool
    :return: bool True if the two input objects are equivalent, False otherwise
    """
    if (vulnerability_1["name"], vulnerability_1["severity"]) == (vulnerability_2["name"], vulnerability_2["severity"]):
        # Do not compare package_version, because this may have been obtained at the time the CVE was first observed
        # on the ECR Scan, which would result in unrelated version updates causing a mismatch while the CVE still
        # applies on both vulnerabilities.
        if all(
            attribute in vulnerability_2["attributes"]
            for attribute in vulnerability_1["attributes"]
            if not attribute["key"] == "package_version"
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


def get_ecr_scan_allowlist_path(image_uri):
    dockerfile_location = test_utils.get_dockerfile_path_for_image(image_uri)
    image_scan_allowlist_path = dockerfile_location + ".os_scan_allowlist.json"
    # Each example image (tied to CUDA version/OS version/other variants) can have its own list of vulnerabilities,
    # which means that we cannot have just a single allowlist for all example images for any framework version.
    if "example" in image_uri:
        image_scan_allowlist_path = dockerfile_location + ".example.os_scan_allowlist.json"
    return image_scan_allowlist_path
