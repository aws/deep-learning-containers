from abc import abstractmethod
import os
import json
import copy, collections
import boto3
import json
import requests

from invoke import run, Context
from time import sleep, time
from enum import IntEnum
from test import test_utils
from test.test_utils import (
    LOGGER,
    EnhancedJSONEncoder,
    ecr as ecr_utils,
    get_installed_python_packages_with_version,
)
import dataclasses
from dataclasses import dataclass
from typing import Any, List
from packaging.version import Version

from tenacity import (
    retry,
    stop_after_delay,
    wait_random_exponential,
)


@dataclass
class VulnerablePackageDetails:
    """
    VulnerablePackageDetails dataclass is used to represent the "package_details" for
    a single vulnerability in Allowlist format.
    """

    file_path: str
    name: str
    package_manager: str
    version: str
    release: str

    def __init__(
        self,
        name: str,
        version: str,
        release: str = None,
        *args: Any,
        **kwargs: Any,
    ):
        self.file_path = kwargs.get("filePath") or kwargs.get("file_path")
        self.name = name
        self.package_manager = kwargs.get("packageManager") or kwargs.get("package_manager")
        self.version = version
        self.release = release


@dataclass
class AllowListFormatVulnerabilityForEnhancedScan:
    """
    AllowListFormatVulnerabilityForEnhancedScan represents how the data looks for a single vulnerability in the allowlist format.
    The data from the ECR Enhanced Results are deserialized into AllowListFormatVulnerabilityForEnhancedScan dataclass. In
    other words, vulnerabilities from the ecr format are directly deserialized into vulnerabilities in Allowlist
    format using AllowListFormatVulnerabilityForEnhancedScan dataclass.
    """

    description: str
    vulnerability_id: str
    name: str
    package_name: str
    package_details: VulnerablePackageDetails
    remediation: dict
    cvss_v3_score: float
    cvss_v30_score: float
    cvss_v31_score: float
    cvss_v2_score: float
    cvss_v3_severity: str
    source_url: str
    source: str
    severity: str
    status: str
    title: str
    reason_to_ignore: str

    def __init__(
        self,
        description: str,
        remediation: dict,
        severity: str,
        status: str,
        title: str,
        *args: Any,
        **kwargs: Any,
    ):
        self.description = description
        packageVulnerabilityDetails = kwargs.get("packageVulnerabilityDetails")
        self.vulnerability_id = (
            packageVulnerabilityDetails["vulnerabilityId"]
            if packageVulnerabilityDetails
            else kwargs["vulnerability_id"]
        )
        self.name = (
            packageVulnerabilityDetails["vulnerabilityId"]
            if packageVulnerabilityDetails
            else kwargs["name"]
        )
        self.package_name = None if packageVulnerabilityDetails else kwargs["package_name"]
        self.package_details = (
            None
            if packageVulnerabilityDetails
            else VulnerablePackageDetails(**kwargs["package_details"])
        )
        self.remediation = remediation
        self.source_url = (
            packageVulnerabilityDetails["sourceUrl"]
            if packageVulnerabilityDetails
            else kwargs["source_url"]
        )
        self.source = (
            packageVulnerabilityDetails["source"]
            if packageVulnerabilityDetails
            else kwargs["source"]
        )
        self.severity = severity
        self.status = status
        self.title = title
        self.cvss_v30_score = (
            self.get_cvss_score(packageVulnerabilityDetails, score_version="3.0")
            if packageVulnerabilityDetails
            else kwargs["cvss_v30_score"]
        )
        self.cvss_v31_score = (
            self.get_cvss_score(packageVulnerabilityDetails, score_version="3.1")
            if packageVulnerabilityDetails
            else kwargs["cvss_v31_score"]
        )
        self.cvss_v3_score = self.cvss_v31_score if self.cvss_v31_score > 0 else self.cvss_v30_score
        self.cvss_v2_score = (
            self.get_cvss_score(packageVulnerabilityDetails, score_version="2.0")
            if packageVulnerabilityDetails
            else kwargs["cvss_v2_score"]
        )
        self.cvss_v3_severity = (
            self.get_cvss_v3_severity(self.cvss_v3_score)
            if packageVulnerabilityDetails
            else kwargs["cvss_v3_severity"]
        )
        self.reason_to_ignore = kwargs.get("reason_to_ignore", "N/A")

    def __eq__(self, other):
        assert type(self) == type(other), f"Types {type(self)} and {type(other)} mismatch!!"
        ## Ignore version key in package_details as it might represent the version of the package existing in the image
        ## and might differ from  image to image, even when the vulnerability is same.
        ## Also ignore the title key of the vulnerablitiy, because, sometimes, 1 vulnerability impacts multiple packages.
        ## In that case, the title key is generated by ECR scans by mentioning the name of all packages in a random order. This fails during comparison.
        if test_utils.check_if_two_dictionaries_are_equal(
            dataclasses.asdict(self.package_details),
            dataclasses.asdict(other.package_details),
            ignore_keys=["version"],
        ):
            return test_utils.check_if_two_dictionaries_are_equal(
                dataclasses.asdict(self),
                dataclasses.asdict(other),
                ignore_keys=["package_details", "title", "reason_to_ignore"],
            )
        return False

    def get_cvss_score(self, packageVulnerabilityDetails: dict, score_version: str = "3.1"):
        """
        The ECR Enhanced Scan returns the CVSS scores as a list under packageVulnerabilityDetails["cvss"].
        The list looks like:
            "packageVulnerabilityDetails": {
                "cvss": [
                    {
                        "baseScore": 7.7,
                        "scoringVector": "CVSS:3.1/AV:N/AC:H/PR:H/UI:N/S:C/C:H/I:N/A:H",
                        "source": "SNYK",
                        "version": "3.1"
                    },
                    {
                        "baseScore": 6.5,
                        "scoringVector": "CVSS:2.0/AV:N/AC:H/PR:H/UI:N/.....",
                        "source": "SNYK",
                        "version": "2.0"
                    }
                ]
            }
        This method iterates through all the CVSS scores and returns the baseScore for a particular CVSS version.
        :param packageVulnerabilityDetails: dict, as described above
        :param score_version: str, desired CVSS version
        :return: float, CVSS score
        """
        for cvss_score in packageVulnerabilityDetails["cvss"]:
            if cvss_score["version"] == score_version:
                return float(cvss_score["baseScore"])
        return 0.0

    ## Taken from https://nvd.nist.gov/vuln-metrics/cvss and section 5 of first.org/cvss/specification-document
    def get_cvss_v3_severity(self, cvss_v3_score: float):
        if cvss_v3_score >= 9.0:
            return "CRITICAL"
        elif cvss_v3_score >= 7.0:
            return "HIGH"
        elif cvss_v3_score >= 4.0:
            return "MEDIUM"
        elif cvss_v3_score >= 0.1:
            return "LOW"
        return "UNDEFINED"  # Used to represent None Severity as well

    def set_package_details_and_name(self, package_details: VulnerablePackageDetails):
        self.package_details = package_details
        self.package_name = self.package_details.name


class ECRScanFailureException(Exception):
    """
    Base class for other exceptions
    """

    pass


class CVESeverity(IntEnum):
    UNTRIAGED = 0
    UNDEFINED = 0
    INFORMATIONAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


class ScanVulnerabilityList:
    """
    ScanVulnerabilityList is a class that reads and stores a vulnerability list in the Allowlist format. The format in which
    the allowlist JSON files are stored on the DLC repo is referred as the Allowlist Format. This class allows easy comparison
    of 2 Allowlist formatted vulnerability lists and defines methods to convert ECR Scan Lists to Allowlist Format lists that
    can be stored within the class itself.
    """

    def __init__(self, minimum_severity=CVESeverity["MEDIUM"]):
        self.vulnerability_list = {}
        self.minimum_severity = minimum_severity

    @abstractmethod
    def are_vulnerabilities_equivalent(self, vulnerability_1, vulnerability_2):
        pass

    @abstractmethod
    def get_vulnerability_package_name_from_allowlist_formatted_vulnerability(self, vulnerability):
        pass

    @abstractmethod
    def construct_allowlist_from_allowlist_formatted_vulnerabilities(
        self, allowlist_formatted_vulnerability_list
    ):
        pass

    def get_flattened_vulnerability_list(self):
        """
        Returns the vulnerability list in the flattened format. For eg., if a vulnerability list looks like
        {"k1":[{"a":"b"},{"c":"d"}], "k2":[{"e":"f"},{"g":"h"}]}, it would return the following:
        [{"a":"b"},{"c":"d"},{"e":"f"},{"g":"h"}]

        :return: List(dict)
        """
        if self.vulnerability_list:
            return [
                vulnerability
                for package_vulnerabilities in self.vulnerability_list.values()
                for vulnerability in package_vulnerabilities
            ]
        return []

    def get_sorted_vulnerability_list(self):
        """
        This method is specifically made to sort the vulnerability list which is actually a dict
        and has the following structure:
        {
            "packge_name1":[
                {"name":"cve-id1", "uri":"http.." ..},
                {"name":"cve-id2", "uri":"http.." ..}
            ],
            "packge_name2":[
                {"name":"cve-id1", "uri":"http.." ..},
                {"name":"cve-id2", "uri":"http.." ..}
            ]
        }
        We want to first sort the innermost list of dicts based on the "name" of each dict and then we sort the
        outermost dict based on keys i.e. package_name1 and package_name2.
        Note: We do not change the actual vulnerability list.
        :return: dict, sorted vulnerability list
        """
        copy_dict = copy.deepcopy(self.vulnerability_list)
        for key, list_of_complex_types in copy_dict.items():
            uniquified_list = test_utils.uniquify_list_of_complex_datatypes(list_of_complex_types)
            uniquified_list.sort(
                key=lambda dict_element: dict_element["name"]
                if isinstance(dict_element, dict)
                else dict_element.name
            )
        return dict(sorted(copy_dict.items()))

    def save_vulnerability_list(self, path):
        if self.vulnerability_list:
            sorted_vulnerability_list = self.get_sorted_vulnerability_list()
            with open(path, "w") as f:
                json.dump(sorted_vulnerability_list, f, indent=4)
        else:
            raise ValueError("self.vulnerability_list is empty.")

    def __contains__(self, vulnerability):
        """
        Check if an input vulnerability exists on the allow-list

        :param vulnerability: dict JSON object consisting of information about the vulnerability in the format
                              presented by the ECR Scan Tool
        :return: bool True if the vulnerability is allowed on the allow-list.
        """
        package_name = self.get_vulnerability_package_name_from_allowlist_formatted_vulnerability(
            vulnerability
        )
        if package_name not in self.vulnerability_list:
            return False
        for allowed_vulnerability in self.vulnerability_list[package_name]:
            if self.are_vulnerabilities_equivalent(vulnerability, allowed_vulnerability):
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
            if len(self.vulnerability_list[package_name]) != len(
                other.vulnerability_list[package_name]
            ):
                return False
            for v1, v2 in zip(
                self.get_sorted_vulnerability_list()[package_name],
                other.get_sorted_vulnerability_list()[package_name],
            ):
                if not self.are_vulnerabilities_equivalent(v1, v2):
                    return False
        return True

    def __eq__(self, other):
        """
        Compare two ScanVulnerabilityList objects for equivalence.

        :param other: Another ScanVulnerabilityList object
        :return: True if equivalent, False otherwise
        """
        return self.__cmp__(other)

    def __ne__(self, other):
        """
        Reverse of __eq__

        :param other: Another ScanVulnerabilityList object
        :return: True if not equivalent, False otherwise
        """
        return not self.__eq__(other)

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

        difference = type(self)(minimum_severity=self.minimum_severity)
        difference.construct_allowlist_from_allowlist_formatted_vulnerabilities(
            missing_vulnerabilities
        )
        return difference

    def __add__(self, other):
        """
        Does Union between ScanVulnerabilityList objects

        :param other: Another ScanVulnerabilityList object
        :return: Union of vulnerabilites exisiting in self and other
        """
        flattened_vulnerability_list_self = self.get_flattened_vulnerability_list()
        flattened_vulnerability_list_other = other.get_flattened_vulnerability_list()
        all_vulnerabilities = flattened_vulnerability_list_self + flattened_vulnerability_list_other
        if not all_vulnerabilities:
            return None
        union_vulnerabilities = test_utils.uniquify_list_of_complex_datatypes(all_vulnerabilities)

        union = type(self)(minimum_severity=self.minimum_severity)
        union.construct_allowlist_from_allowlist_formatted_vulnerabilities(union_vulnerabilities)
        return union


class ECRBasicScanVulnerabilityList(ScanVulnerabilityList):
    """
    A child class of ScanVulnerabilityList that is specifically made to deal with ECR Basic Scans.
    """

    def get_vulnerability_package_name_from_allowlist_formatted_vulnerability(self, vulnerability):
        """
        Get Package Name from a vulnerability JSON object.
        For ECR Basic Scans, the format of the vulnerability is same in ecr format and allowlist format, so this function
        can be used interchangeably.

        :param vulnerability: dict JSON object consisting of information about the vulnerability in the Allowlist format data
        which is same as ECR Scan Tool data for ECR Basic Scanning.
        :return: str package name
        """
        for attribute in vulnerability["attributes"]:
            if attribute["key"] == "package_name":
                return attribute["value"]
        return None

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

    def construct_allowlist_from_allowlist_formatted_vulnerabilities(
        self, allowlist_formatted_vulnerability_list
    ):
        """
        Read a vulnerability list and construct the vulnerability_list

        :param vulnerability_list: list ECR Scan Result results
        :return: dict self.vulnerability_list
        """
        for vulnerability in allowlist_formatted_vulnerability_list:
            package_name = (
                self.get_vulnerability_package_name_from_allowlist_formatted_vulnerability(
                    vulnerability
                )
            )
            if package_name not in self.vulnerability_list:
                self.vulnerability_list[package_name] = []
            if CVESeverity[vulnerability["severity"]] >= self.minimum_severity:
                self.vulnerability_list[package_name].append(vulnerability)
        return self.vulnerability_list

    def construct_allowlist_from_ecr_scan_result(self, ecr_format_vulnerability_list):
        """
        Read a vulnerability list and construct the vulnerability_list
        For Basic Scan, the ecr scan vulnerabilities and the allowlist vulnerabilities have the same format
        and hence we can use the same function.

        :param vulnerability_list: list ECR Scan Result results
        :return: dict self.vulnerability_list
        """
        return self.construct_allowlist_from_allowlist_formatted_vulnerabilities(
            ecr_format_vulnerability_list
        )

    def are_vulnerabilities_equivalent(self, vulnerability_1, vulnerability_2):
        """
        Check if two vulnerability JSON objects are equivalent

        :param vulnerability_1: dict JSON object consisting of information about the vulnerability in the format
                                presented by the ECR Scan Tool
        :param vulnerability_2: dict JSON object consisting of information about the vulnerability in the format
                                presented by the ECR Scan Tool
        :return: bool True if the two input objects are equivalent, False otherwise
        """
        if (vulnerability_1["name"], vulnerability_1["severity"]) == (
            vulnerability_2["name"],
            vulnerability_2["severity"],
        ):
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


class ECREnhancedScanVulnerabilityList(ScanVulnerabilityList):
    """
    A child class of ScanVulnerabilityList that is specifically made to deal with ECR Enhanced Scans.
    """

    def get_vulnerability_package_name_from_allowlist_formatted_vulnerability(
        self, vulnerability: AllowListFormatVulnerabilityForEnhancedScan
    ):
        """
        Get Package Name from a vulnerability JSON object
        :param vulnerability: dict JSON object consisting of information about the vulnerability in the Allowlist Format.
        :return: str package name
        """
        return vulnerability.package_name

    def construct_allowlist_from_file(self, file_path):
        """
        Read JSON file that has the vulnerability data saved in the Allowlist format itself and prepare the object with
        all the vulnerabilities in the Allowlist format as well.

        :param file_path: Path to the allow-list JSON file.
        :return: dict self.vulnerability_list
        """
        with open(file_path, "r") as f:
            file_allowlist = json.load(f)
        for _, package_vulnerability_list in file_allowlist.items():
            allowlist_formatted_package_vulnerability_list = [
                AllowListFormatVulnerabilityForEnhancedScan(**vulnerability)
                for vulnerability in package_vulnerability_list
            ]
            self.construct_allowlist_from_allowlist_formatted_vulnerabilities(
                allowlist_formatted_package_vulnerability_list
            )
        return self.vulnerability_list

    def construct_allowlist_from_allowlist_formatted_vulnerabilities(
        self,
        allowlist_formatted_vulnerability_list: List[AllowListFormatVulnerabilityForEnhancedScan],
    ):
        """
        Read a vulnerability list in the AllowListFormat and construct the vulnerability_list in the same format.

        :param vulnerability_list: list ECR Scan Result results
        :return: dict self.vulnerability_list
        """
        for vulnerability in allowlist_formatted_vulnerability_list:
            package_name = (
                self.get_vulnerability_package_name_from_allowlist_formatted_vulnerability(
                    vulnerability
                )
            )
            if CVESeverity[vulnerability.cvss_v3_severity] < self.minimum_severity:
                continue
            if package_name not in self.vulnerability_list:
                self.vulnerability_list[package_name] = []
            self.vulnerability_list[package_name].append(vulnerability)
        return self.vulnerability_list

    def allow_vendor_severity_override(self, vulnerability_obj):
        """
        If package source is from an allowed vendor, allow the vendor's severity to take precedence
        Args:
            vulnerability_obj (AllowListFormatVulnerabilityForEnhancedScan): object representing the vulnerability
        Return:
            bool: Whether to allow the vulnerability or not
        """
        allowed_vendors = {"UBUNTU_CVE"}
        return (
            vulnerability_obj.source in allowed_vendors
            and CVESeverity[vulnerability_obj.severity] < self.minimum_severity
        )

    def allow_cvss_v3_severity(self, vulnerability_obj):
        """
        If CVSS v3 score is less than the threshold, return True, else return False
        Args:
            vulnerability_obj (AllowListFormatVulnerabilityForEnhancedScan): object representing the vulnerability
        Return:
            bool: Whether to allow the vulnerablity or not
        """
        return CVESeverity[vulnerability_obj.cvss_v3_severity] < self.minimum_severity

    def construct_allowlist_from_ecr_scan_result(self, ecr_format_vulnerability_list):
        """
        Read an ECR formatted vulnerability list and construct the Allowlist Formatted vulnerability_list

        :param vulnerability_list: list ECR Scan Result results
        :return: dict self.vulnerability_list
        """
        for ecr_format_vulnerability in ecr_format_vulnerability_list:
            for vulnerable_package in ecr_format_vulnerability["packageVulnerabilityDetails"][
                "vulnerablePackages"
            ]:
                allowlist_format_vulnerability_object = AllowListFormatVulnerabilityForEnhancedScan(
                    **ecr_format_vulnerability
                )
                vulnerable_package_object = VulnerablePackageDetails(**vulnerable_package)
                allowlist_format_vulnerability_object.set_package_details_and_name(
                    vulnerable_package_object
                )
                if self.allow_cvss_v3_severity(
                    allowlist_format_vulnerability_object
                ) or self.allow_vendor_severity_override(allowlist_format_vulnerability_object):
                    continue
                if (
                    allowlist_format_vulnerability_object.package_name
                    not in self.vulnerability_list
                ):
                    self.vulnerability_list[allowlist_format_vulnerability_object.package_name] = []
                self.vulnerability_list[allowlist_format_vulnerability_object.package_name].append(
                    allowlist_format_vulnerability_object
                )
        self.vulnerability_list = self.get_sorted_vulnerability_list()
        return self.vulnerability_list

    def are_vulnerabilities_equivalent(self, vulnerability_1, vulnerability_2):
        """
        Check if two vulnerability JSON objects are equivalent

        :param vulnerability_1: dict, JSON object consisting of information about the vulnerability in the Allowlist Format
        :param vulnerability_2: dict, JSON object consisting of information about the vulnerability in the Allowlist Format
        :return: bool True if the two input objects are equivalent, False otherwise
        """
        return vulnerability_1 == vulnerability_2

    def get_summarized_info(self):
        """
        Gets summarized info regarding all the packages vulnerability_list and all the vulenrability IDs corresponding to them.
        """
        summarized_list = []
        for package_name, vulnerabilities in self.vulnerability_list.items():
            for vulnerability in vulnerabilities:
                summarized_list.append(
                    (package_name, vulnerability.vulnerability_id, vulnerability.severity)
                )
        summarized_list = sorted(list(set(summarized_list)))
        return summarized_list


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


def _save_lists_in_s3(save_details, s3_bucket_name):
    """
    This method takes in a list of filenames and the data corresponding to each filename and stores it in
    the s3 bucket.

    :param save_details: list[(string, list)], a lists of tuples wherein each tuple has a filename and the corresponding data.
    :param s3_bucket_name: string, name of the s3 bucket
    """
    s3_client = boto3.client("s3")
    for filename, data in save_details:
        with open(filename, "w") as outfile:
            json.dump(data, outfile, indent=4)
        s3_client.upload_file(Filename=filename, Bucket=s3_bucket_name, Key=filename)


def get_target_image_uri_using_current_uri_and_target_repo(
    image, target_repository_name, target_repository_region, append_tag=""
):
    """
    This function helps formulate a target image uri for a given image such that the target uri retains
    the old uri info (i.e. old repo name and old repo tag).

    :param image: str, image uri
    :param target_repository_name: str, name of target repository
    :param target_repository_region: str, region of target repository
    :param append_tag: str, string that needs to be appended at the end of the tag
    :return: str, target image uri
    """
    sts_client = boto3.client("sts", region_name=target_repository_region)
    account_id = sts_client.get_caller_identity().get("Account")
    registry = ecr_utils.get_ecr_registry(account_id, target_repository_region)
    (
        original_image_repository,
        original_image_tag,
    ) = test_utils.get_repository_and_tag_from_image_uri(image)
    if append_tag:
        upgraded_image_tag = f"{original_image_repository}-{original_image_tag}-{append_tag}"
    else:
        upgraded_image_tag = f"{original_image_repository}-{original_image_tag}"
    target_image_uri = f"{registry}/{target_repository_name}:{upgraded_image_tag}"
    return target_image_uri


def run_upgrade_on_image_and_push(image, new_image_uri):
    """
    Creates a container for the image being tested. Runs apt update and upgrade on the container
    and the commits the container as new_image_uri. This new image is then pushed to the ECR.

    :param image: str
    :param new_image_uri: str
    """
    max_attempts = 10
    ctx = Context()
    docker_run_cmd = f"docker run -id --entrypoint='/bin/bash' {image}"
    container_id = ctx.run(f"{docker_run_cmd}", hide=True).stdout.strip()
    apt_command = "apt-get update && apt-get upgrade"
    docker_exec_cmd = f"docker exec -i {container_id}"
    attempt_count = 0
    apt_ran_successfully_flag = False
    # When a command or application is updating the system or installing a new software, it locks the dpkg file (Debian package manager).
    # Since we have multiple processes running for the tests, there are cases when one of the process locks the dpkg file
    # In this scenario, we get error: ‘E: Could not get lock /var/lib/dpkg/lock’ while running apt-get update
    # That is why we need multiple tries to ensure that it succeeds in one of the tries.
    # More info: https://itsfoss.com/could-not-get-lock-error/
    while True:
        run_output = ctx.run(f"{docker_exec_cmd} {apt_command}", hide=True, warn=True)
        attempt_count += 1
        if not run_output.ok:
            test_utils.LOGGER.info(
                f"Attempt no. {attempt_count} on image: {image}"
                f"Could not run apt update and upgrade. \n"
                f"Stdout is {run_output.stdout} \n"
                f"Stderr is {run_output.stderr} \n"
                f"Failed status is {run_output.exited}"
            )
            sleep(2 * 60)
        elif run_output.ok:
            apt_ran_successfully_flag = True
            break
        if attempt_count == max_attempts:
            break
    if not apt_ran_successfully_flag:
        raise RuntimeError(
            f"Could not run apt update and upgrade on image: {image}. \n"
            f"Stdout is {run_output.stdout} \n"
            f"Stderr is {run_output.stderr} \n"
            f"Failed status is {run_output.exited}"
        )
    ctx.run(f"docker commit {container_id} {new_image_uri}", hide=True)
    ctx.run(f"docker rm -f {container_id}", hide=True)
    ctx.run(f"docker push {new_image_uri}", hide=True)


def _invoke_lambda(function_name, payload_dict={}):
    """
    Asyncronously Invokes the passed lambda.

    :param function_name: str, name of the lambda function
    :param payload_dict: dict, payload to be sent to the lambda
    """
    lambda_client = boto3.client("lambda", region_name=test_utils.DEFAULT_REGION)
    response = lambda_client.invoke(
        FunctionName=function_name,
        InvocationType="Event",
        LogType="Tail",
        Payload=json.dumps(payload_dict),
    )
    status_code = response.get("StatusCode")
    if status_code != 202:
        raise ValueError("Lambda call not made properly. Status code returned {status_code}")


def get_apt_package_name(ecr_package_name):
    """
    Few packages have different names in the ecr scan and actual apt. This function returns an
    apt name of an ecr package.
    :param ecr_package_name: str, name of the package in ecr scans
    :param apt_package_name: str, name of the package in apt
    """
    name_mapper = {
        "cyrus-sasl2": "libsasl2-2",
        "glibc": "libc6",
        "libopenmpt": "libopenmpt-dev",
        "fribidi": "libfribidi-dev",
    }
    return name_mapper.get(ecr_package_name, ecr_package_name)


def create_and_save_package_list_to_s3(old_filepath, new_packages, new_filepath, s3_bucket_name):
    """
    This method conducts the union of packages present in the original apt-get-upgrade
    list and new list of packages passed as an argument. It makes a new file and stores
    the results in it.
    :param old_filpath: str, path of original file
    :param new_packages: list[str], consists of list of packages
    :param new_filpath: str, path of new file that will have the results of union
    :param s3_bucket_name: string, name of the s3 bucket
    """
    file1 = open(old_filepath, "r")
    lines = file1.readlines()
    current_packages = [line.strip() for line in lines]
    package_list = current_packages
    new_packages = [get_apt_package_name(new_package) for new_package in new_packages]
    union_of_old_and_new_packages = set(package_list).union(set(new_packages))
    unified_package_list = list(union_of_old_and_new_packages)
    unified_package_list.sort()
    unified_package_list_for_storage = [
        f"{package_name}\n" for package_name in unified_package_list
    ]
    file1.close()
    run(f"rm -rf {new_filepath}")
    with open(new_filepath, "w") as file2:
        file2.writelines(unified_package_list_for_storage)
    s3_client = boto3.client("s3")
    s3_client.upload_file(Filename=new_filepath, Bucket=s3_bucket_name, Key=new_filepath)


def save_scan_vulnerability_list_object_to_s3_in_json_format(
    image, scan_vulnerability_list_object, append_tag, s3_bucket_name
):
    """
    Saves the vulnerability list in the s3 bucket. It uses image to decide the name of the file on
    the s3 bucket.

    :param image: str, image uri
    :param vulnerability_list: ScanVulnerabilityList
    :param s3_bucket_name: string, name of the s3 bucket
    :return: str, name of the file as stored on s3
    """
    processed_image_uri = image.replace(".", "-").replace("/", "-").replace(":", "-")
    file_name = f"{processed_image_uri}-{append_tag}.json"
    scan_vulnerability_list_object.save_vulnerability_list(file_name)
    s3_client = boto3.client("s3")
    s3_client.upload_file(Filename=file_name, Bucket=s3_bucket_name, Key=file_name)
    return file_name


def get_vulnerabilites_fixable_by_upgrade(
    image_allowlist, ecr_image_vulnerability_list, upgraded_image_vulnerability_list
):
    """
    Finds out the vulnerabilities that are fixable by apt-get update and apt-get upgrade.

    :param image_allowlist: ScanVulnerabilityList, Vulnerabities that are present in the respective allowlist in the DLC git repo.
    :param ecr_image_vulnerability_list: ScanVulnerabilityList, Vulnerabities recently detected WITHOUT running apt-upgrade on the originally released image.
    :param upgraded_image_vulnerability_list: ScanVulnerabilityList, Vulnerabilites exisiting in the image WITH apt-upgrade run on it.
    :return: ScanVulnerabilityList/NONE, either ScanVulnerabilityList object or None if no fixable vulnerability
    """
    fixable_ecr_image_scan_vulnerabilites = (
        ecr_image_vulnerability_list - upgraded_image_vulnerability_list
    )
    fixable_allowlist_vulnerabilites = image_allowlist - upgraded_image_vulnerability_list
    vulnerabilities_fixable_by_upgrade = None
    if fixable_ecr_image_scan_vulnerabilites and fixable_allowlist_vulnerabilites:
        vulnerabilities_fixable_by_upgrade = (
            fixable_ecr_image_scan_vulnerabilites + fixable_allowlist_vulnerabilites
        )
    elif fixable_ecr_image_scan_vulnerabilites:
        vulnerabilities_fixable_by_upgrade = fixable_ecr_image_scan_vulnerabilites
    elif fixable_allowlist_vulnerabilites:
        vulnerabilities_fixable_by_upgrade = fixable_allowlist_vulnerabilites
    return vulnerabilities_fixable_by_upgrade


def conduct_failure_routine(
    image,
    image_allowlist,
    ecr_image_vulnerability_list,
    upgraded_image_vulnerability_list,
    s3_bucket_for_storage,
):
    """
    This method conducts the entire process that is supposed to be followed when ECR test fails. It finds all
    the fixable and non fixable vulnerabilities and all the packages that can be upgraded and finally invokes
    the Auto-Secure lambda for further processing.

    :param image: str, image uri
    :param image_allowlist: ScanVulnerabilityList, Vulnerabities that are present in the respective allowlist in the DLC git repo.
    :param ecr_image_vulnerability_list: ScanVulnerabilityList, Vulnerabities recently detected WITHOUT running apt-upgrade on the originally released image.
    :param upgraded_image_vulnerability_list: ScanVulnerabilityList, Vulnerabilites exisiting in the image WITH apt-upgrade run on it.
    :param s3_bucket_for_storage: s3 name of the bucket that would be used for saving all the important data that needs to be stored during failure routine.
    :return: dict, a dictionary consisting of the entire summary of the steps run within this method.
    """
    s3_filename_for_allowlist = save_scan_vulnerability_list_object_to_s3_in_json_format(
        image, upgraded_image_vulnerability_list, "allowlist", s3_bucket_for_storage
    )
    s3_filename_for_current_image_ecr_scan_list = (
        save_scan_vulnerability_list_object_to_s3_in_json_format(
            image, ecr_image_vulnerability_list, "current-ecr-scanlist", s3_bucket_for_storage
        )
    )
    original_filepath_for_allowlist = test_utils.get_ecr_scan_allowlist_path(image)
    edited_files = [
        {
            "s3_filename": s3_filename_for_allowlist,
            "github_filepath": original_filepath_for_allowlist,
        }
    ]
    vulnerabilities_fixable_by_upgrade = get_vulnerabilites_fixable_by_upgrade(
        image_allowlist, ecr_image_vulnerability_list, upgraded_image_vulnerability_list
    )
    newly_found_non_fixable_vulnerabilites = upgraded_image_vulnerability_list - image_allowlist
    fixable_list = {}
    if vulnerabilities_fixable_by_upgrade:
        fixable_list = vulnerabilities_fixable_by_upgrade.vulnerability_list
    apt_upgrade_list_filename = (
        f"apt-upgrade-list-{test_utils.get_processor_from_image_uri(image)}.txt"
    )
    s3_filename_for_apt_upgrade_list = s3_filename_for_allowlist.replace(
        "allowlist.json", apt_upgrade_list_filename
    )
    original_filepath_for_apt_upgrade_list = os.path.join(
        os.path.dirname(original_filepath_for_allowlist), apt_upgrade_list_filename
    )
    new_package_list = fixable_list if isinstance(fixable_list, list) else list(fixable_list.keys())
    create_and_save_package_list_to_s3(
        original_filepath_for_apt_upgrade_list,
        new_package_list,
        s3_filename_for_apt_upgrade_list,
        s3_bucket_for_storage,
    )
    edited_files.append(
        {
            "s3_filename": s3_filename_for_apt_upgrade_list,
            "github_filepath": original_filepath_for_apt_upgrade_list,
        }
    )
    newly_found_non_fixable_list = {}
    if newly_found_non_fixable_vulnerabilites:
        newly_found_non_fixable_list = newly_found_non_fixable_vulnerabilites.vulnerability_list
    message_body = {
        "edited_files": edited_files,
        "fixable_vulnerabilities": fixable_list,
        "non_fixable_vulnerabilities": newly_found_non_fixable_list,
    }
    ## TODO: Make the conditions below as if test_utils.is_canary_context() and test_utils.is_time_for_invoking_ecr_scan_failure_routine_lambda() and os.getenv("REGION") == test_utils.DEFAULT_REGION:
    ## to make sure that we just invoke the ECR_SCAN_FAILURE_ROUTINE_LAMBDA once everyday
    if test_utils.is_canary_context() and os.getenv("REGION") == test_utils.DEFAULT_REGION:
        # boto3.Session().region_name == test_utils.DEFAULT_REGION helps us invoke the ECR_SCAN_FAILURE_ROUTINE_LAMBDA
        # from just 1 account
        _invoke_lambda(
            function_name=test_utils.ECR_SCAN_FAILURE_ROUTINE_LAMBDA, payload_dict=message_body
        )
    return_dict = copy.deepcopy(message_body)
    return_dict["s3_filename_for_allowlist"] = s3_filename_for_allowlist
    return_dict[
        "s3_filename_for_current_image_ecr_scan_list"
    ] = s3_filename_for_current_image_ecr_scan_list
    return return_dict


def process_failure_routine_summary_and_store_data_in_s3(failure_routine_summary, s3_bucket_name):
    """
    This method is especially constructed to process the failure routine summary that is generated as a result of
    calling conduct_failure_routine. It extracts lists and calls the save lists function to store them in the s3
    bucket.

    :param failure_routine_summary: dict, dictionary returned as an outcome of conduct_failure_routine method
    :param s3_bucket_name: string, name of the s3 bucket
    :return s3_filename_for_fixable_list: string, filename in the s3 bucket for the fixable vulnerabilities
    :return s3_filename_for_non_fixable_list: string, filename in the s3 bucket for the non-fixable vulnerabilities
    """
    s3_filename_for_allowlist = failure_routine_summary["s3_filename_for_allowlist"]
    s3_filename_for_fixable_list = s3_filename_for_allowlist.replace(
        "allowlist.json", "fixable-vulnerability-list.json"
    )
    s3_filename_for_non_fixable_list = s3_filename_for_allowlist.replace(
        "allowlist.json", "non-fixable-vulnerability-list.json"
    )
    save_details = []
    save_details.append(
        (s3_filename_for_fixable_list, failure_routine_summary["fixable_vulnerabilities"])
    )
    save_details.append(
        (s3_filename_for_non_fixable_list, failure_routine_summary["non_fixable_vulnerabilities"])
    )
    _save_lists_in_s3(save_details, s3_bucket_name)
    return s3_filename_for_fixable_list, s3_filename_for_non_fixable_list


def run_scan(ecr_client, image):
    scan_status = None
    start_time = time()
    ecr_utils.start_ecr_image_scan(ecr_client, image)
    while (time() - start_time) <= 600:
        scan_status, scan_status_description = ecr_utils.get_ecr_image_scan_status(
            ecr_client, image
        )
        if scan_status == "FAILED" or scan_status not in [None, "IN_PROGRESS", "COMPLETE"]:
            raise ECRScanFailureException(
                f"ECR Scan failed for {image} with description: {scan_status_description}"
            )
        if scan_status == "COMPLETE":
            break
        sleep(1)
    if scan_status != "COMPLETE":
        raise TimeoutError(f"ECR Scan is still in {scan_status} state. Exiting.")


def wait_for_enhanced_scans_to_complete(ecr_client, image):
    """
    For Continuous Enhanced scans, the images will go through `SCAN_ON_PUSH` when they are uploaded for the
    first time. During that time, their state will be shown as `PENDING`. From next time onwards, their status will show
    itself as `ACTIVE`.

    :param ecr_client: boto3 Client for ECR
    :param image: str, Image URI for image being scanned
    """
    scan_status = None
    scan_status_description = ""
    start_time = time()
    while (time() - start_time) <= 45 * 60:
        try:
            scan_status, scan_status_description = ecr_utils.get_ecr_image_enhanced_scan_status(
                ecr_client, image
            )
        except ecr_client.exceptions.ScanNotFoundException as e:
            LOGGER.info(e.response)
            LOGGER.info(
                "It takes sometime for the newly uploaded image to show its scan status, hence the error handling"
            )
        if scan_status == "ACTIVE":
            break
        sleep(1 * 60)
    if scan_status != "ACTIVE":
        raise TimeoutError(
            f"ECR Scan is still in {scan_status} state with description: {scan_status_description}. Exiting."
        )


def fetch_other_vulnerability_lists(image, ecr_client, minimum_sev_threshold):
    """
    For a given image it fetches all the other vulnerability lists except the vulnerability list formed by the
    ecr scan of the current image. In other words, for a given image it fetches upgraded_image_vulnerability_list and
    image_scan_allowlist.

    :param image: str Image URI for image to be tested
    :param ecr_client: boto3 Client for ECR
    :param minimum_sev_threshold: string, determines the minimum severity threshold for ScanVulnerabilityList objects. Can take values HIGH or MEDIUM.
    :return upgraded_image_vulnerability_list: ScanVulnerabilityList, Vulnerabilites exisiting in the image WITH apt-upgrade run on it.
    :return image_allowlist: ScanVulnerabilityList, Vulnerabities that are present in the respective allowlist in the DLC git repo.
    """
    new_image_uri_for_upgraded_image = get_target_image_uri_using_current_uri_and_target_repo(
        image,
        target_repository_name=test_utils.UPGRADE_ECR_REPO_NAME,
        target_repository_region=os.getenv("REGION", test_utils.DEFAULT_REGION),
        append_tag="upgraded",
    )
    run_upgrade_on_image_and_push(image, new_image_uri_for_upgraded_image)
    run_scan(ecr_client, new_image_uri_for_upgraded_image)
    scan_results_with_upgrade = ecr_utils.get_ecr_image_scan_results(
        ecr_client, new_image_uri_for_upgraded_image, minimum_vulnerability=minimum_sev_threshold
    )
    scan_results_with_upgrade = ecr_utils.populate_ecr_scan_with_web_scraper_results(
        new_image_uri_for_upgraded_image, scan_results_with_upgrade
    )
    upgraded_image_vulnerability_list = ECRBasicScanVulnerabilityList(
        minimum_severity=CVESeverity[minimum_sev_threshold]
    )
    upgraded_image_vulnerability_list.construct_allowlist_from_ecr_scan_result(
        scan_results_with_upgrade
    )
    image_scan_allowlist = ECRBasicScanVulnerabilityList(
        minimum_severity=CVESeverity[minimum_sev_threshold]
    )
    image_scan_allowlist_path = test_utils.get_ecr_scan_allowlist_path(image)
    if os.path.exists(image_scan_allowlist_path):
        image_scan_allowlist.construct_allowlist_from_file(image_scan_allowlist_path)
    return upgraded_image_vulnerability_list, image_scan_allowlist


def generate_future_allowlist(
    ecr_image_vulnerability_list: ECREnhancedScanVulnerabilityList,
    image_scan_allowlist: ECREnhancedScanVulnerabilityList,
    non_patchable_vulnerabilities: ECREnhancedScanVulnerabilityList,
):
    """
    This method helps in generating the future allowlist. It takes 2 vulnerability_list objects as input, namely - ecr_image_vulnerability_list (consists
    of the vulnerabilities found in latest ECR Scan), image_scan_allowlist (consists of the allowlist vulns that exist on git repo) and
    non_patchable_vulnerabilities (consits of the non-patchable vulns that are extract from extract_non_patchable_vulnerabilities).

    1. It finds the old/redundant vulnerabilities that are existing in the allowlist. This is done by removing all the image_scan_allowlist that are not
       shown on the latest scan.
    2. Then, it removes these non-relevant vulns from the image_scan_allowlist and stores this in future_allowlist
    3. In the end, it add the non_patchable vulns to the future_allowlist generated in step 2

    :return: Object of type ECREnhancedScanVulnerabilityList, this is the new/future allowlist that will be used by pr-generator.
    """
    non_relevant_allowlist_vulnerabilities = image_scan_allowlist - ecr_image_vulnerability_list
    future_allowlist = image_scan_allowlist - non_relevant_allowlist_vulnerabilities
    if future_allowlist:
        future_allowlist = future_allowlist + non_patchable_vulnerabilities
    else:
        future_allowlist = copy.deepcopy(non_patchable_vulnerabilities)
    return future_allowlist


def segregate_impacted_package_names_based_on_manager(
    vulnerability_list_object: ECREnhancedScanVulnerabilityList,
):
    """
    This method takes the latest ECREnhancedScanVulnerabilityList and segregates the impacted packages based on their package managers.

    :param vulnerability_list_object: Object of type ECREnhancedScanVulnerabilityList
    :return: Dict[key=package_manager_type, value=set of package names]
    """
    segregated_package_names = {}
    segregated_package_names["os_packages"] = set()
    segregated_package_names["py_packages"] = set()
    for package_name, package_cve_list in vulnerability_list_object.vulnerability_list.items():
        for cve in package_cve_list:
            if cve.package_details.package_manager == "OS":
                segregated_package_names["os_packages"].add(package_name)
            elif cve.package_details.package_manager == "PYTHONPKG":
                segregated_package_names["py_packages"].add(package_name)
    return segregated_package_names


def run_patch_evaluation_script_to_reevaluate_package_status(
    docker_exec_cmd, image_uri, impacted_os_packages
):
    """
    This method runs the miscellaneous_scripts/extract_apt_patch_data.py on the DLC based on the latest impacted packages and returns the generated data.

    :param docker_exec_cmd: str, docker_exec_cmd
    :param image_uri: str, Image URI
    :param impacted_os_packages: set, Consists of the latest impacted packages
    :return new_apt_patch_evaluation_data: dict, this is the data generated by extract_apt_patch_data.py script and looks like
        {
            "patch_package_dict": [List of packages],
            "upgradable_packages_data_for_impacted_packages": Dict[key=source_package_name, value=List of all packages that have the source as key]
        }
    """
    save_file_name = f"""{image_uri.replace("/","_").replace(":","_")}-apt-results.json"""
    ## TODO: Remove
    impacted_os_packages.add("imagemagick")
    script_run_cmd = f"""python /deep-learning-containers/miscellaneous_scripts/extract_apt_patch_data.py --impacted-packages {",".join(list(impacted_os_packages))} --save-result-path /deep-learning-containers/{save_file_name} --mode_type generate"""
    run(f"{docker_exec_cmd} {script_run_cmd}", hide=True)
    new_apt_patch_evaluation_data = {}
    new_apt_patch_evaluation_data_location = os.path.join(
        test_utils.get_repository_local_path(), save_file_name
    )
    if os.path.exists:
        with open(new_apt_patch_evaluation_data_location, "r") as readfile:
            new_apt_patch_evaluation_data = json.load(readfile)
    return new_apt_patch_evaluation_data


def get_os_package_upgradable_status(
    package: str, new_apt_patch_evaluation_data: dict, embedded_apt_patch_evaluation_data: dict
):
    """
    Checks if the OS package is upgradable or not. It uses the new_apt_patch_evaluation_data that has been generated by running the miscellaneous_scripts/extract_apt_patch_data.py on the
    newly built DLC. The new_apt_patch_evaluation_data has the data retrieved on evaluating the DLC against the latest vulnerabilities found in it after patching. It also uses the embedded_apt_patch_evaluation_data
    that was generated on the old DLC during the build time. The embedded_apt_patch_evaluation_data had evaluated the DLC against the vulnerabilities that existed with old package configurations.

    To check upgradability, we see if the package or any of its binaries are existing in the new_apt_patch_evaluation_data["upgradable_packages_data_for_impacted_packages"].
    If they do not exist, then there is no scope for upgrading the package and we declare the package as non-upgradable. While declaring the package as
    non-upgradable, we see if the package or its binaries were upgraded during the build time using embedded_apt_patch_evaluation_data and add that info
    to the message for better information.

    :param pacakge: str, name of package
    :param new_apt_patch_evaluation_data: dict, The new_apt_patch_evaluation_data has the data retrieved on evaluating the DLC against the latest vulnerabilities found in it after patching.
    :param embedded_apt_patch_evaluation_data: dict, The embedded_apt_patch_evaluation_data has the evaluation data of the old DLC against the vulnerabilities that existed with old package configurations.
    :return: [bool, str], returns 2 values, the first says True if the vulnerability/package is non-upgradable and the second one stores the ignore message
             in case the vulnerability is non-upgradable. This ignore message is used to insert into the allowlist.
    """
    is_package_upgradable = True
    ignore_message = ""
    if package not in new_apt_patch_evaluation_data.get(
        "upgradable_packages_data_for_impacted_packages", {}
    ):
        is_package_upgradable = False
        ignore_message = "Package and its binaries cannot be upgraded further."
        if package in embedded_apt_patch_evaluation_data.get(
            "upgradable_packages_data_for_impacted_packages", {}
        ):
            package_related_binaries = sorted(
                embedded_apt_patch_evaluation_data[
                    "upgradable_packages_data_for_impacted_packages"
                ][package]
            )
            ignore_message = f"""{ignore_message} Packages: {",".join(package_related_binaries)} have been upgraded."""
    return is_package_upgradable, ignore_message

@retry(
    reraise=True,
    stop=stop_after_delay(20 * 60),  # Keep retrying for 20 minutes
    wait=wait_random_exponential(min=30, max=2 * 60),  # Retry after waiting 30 secsonds - 2 minutes
)
def get_latest_version_of_a_python_package(package_name:str):
    """
    Get the latest version of a python package. Calls PyPi to extract the same.

    :return: str, version of the package
    """
    response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
    latest_version = response.json()["info"]["version"]
    return latest_version


def check_if_python_vulnerability_is_non_patchable_and_get_ignore_message(
    vulnerability: AllowListFormatVulnerabilityForEnhancedScan,
    installed_python_package_version_dict: dict,
    docker_exec_command: str,
):
    """
    This method takes in the vulnerability Object and checks if it is non-patchable (or allowlistable in other words). It makes the following checks:
    1. Check if the installed package version is same as the one being shown by scanner, if not declare it as non-patchable/allowlistable
    2. Check if the package is already in the latest version, if not declare it as non-patchable/allowlistable

    :param vulnerability: Object of type AllowListFormatVulnerabilityForEnhancedScan, vuln that we need to check patchability for.
    :param installed_python_package_version_dict: Dict, key = package name and value = package version. This is the list of packages
                                                        that are installed in the image being currently tested.
    :param docker_exec_command: str, Docker exec command to run additional commands on the container
    :return: [bool, str], returns 2 values, the first says True if the vulnerability/package is non-patchable and the second one stores the ignore message
             in case the vulnerability is non-patchable. This ignore message is used to insert into the allowlist.
    """
    assert (
        vulnerability.package_details.package_manager == "PYTHONPKG"
    ), f"Vulnerability: {json.dumps(vulnerability, cls=EnhancedJSONEncoder)} is not PythonPkg managed."

    package_name = vulnerability.package_name.lower()
    if package_name not in installed_python_package_version_dict:
        # To begin with, we will not allowlist package names that are not present in the installed list and show up in the vulnerability.
        # However based on the observed behavior, or such scenarios occuring in the future, we will start to allowlist this vulnerability.
        LOGGER.info(f"Package {package_name} not found!")
        return False, ""

    ## 1. Check if the installed package version is same as the one being shown by scanner
    installed_package_version = installed_python_package_version_dict.get(package_name)
    vulnerability_package_version = vulnerability.package_details.version
    if installed_package_version != vulnerability_package_version:
        return (
            True,
            f"Installed package version is {installed_package_version} which is not equal to the one shown in vulnerability.",
        )

    ## 2. Check if the package is already in the latest version, if not then allowlist
    latest_version = get_latest_version_of_a_python_package(package_name=package_name)
    if Version(installed_package_version) == Version(latest_version):
        return True, f"Installed package version {installed_package_version} is the latest version"

    ##TODO: Revert
    if vulnerability.package_name == "gevent":
        return True, "Custom Ignore Message - Trshanta"
    return False, ""


def get_non_patchable_python_vulnerabilities(
    vulnerability_list: List[AllowListFormatVulnerabilityForEnhancedScan],
    installed_python_package_version_dict: dict,
    docker_exec_command: str,
):
    """
    This method looks into all the vulnerabilities associated with a Python package and then iterates through each vulnerability to see
    if it is non-patchable or not. It uses check_if_python_vulnerability_is_non_patchable_and_get_ignore_message method to check the same.
    Thereafter, it returns the list of all the vulns that are non-patchable and can be added to the allowlist.

    :param vulnerability_list: List[AllowListFormatVulnerabilityForEnhancedScan], The list of all the vulns found for a package
    :param installed_python_package_version_dict: dict, Dictionary with package name as keys and their values as version.
    :param docker_exec_command: str, The docker exec command
    :return: List[AllowListFormatVulnerabilityForEnhancedScan], list of all the non-patchable vulns
    """
    non_patchable_list = []
    for vulnerability in vulnerability_list:
        (
            is_python_vulnerability_non_patchable,
            ignore_msg,
        ) = check_if_python_vulnerability_is_non_patchable_and_get_ignore_message(
            vulnerability=vulnerability,
            installed_python_package_version_dict=installed_python_package_version_dict,
            docker_exec_command=docker_exec_command,
        )
        if is_python_vulnerability_non_patchable:
            vulnerability.reason_to_ignore = ignore_msg
            non_patchable_list.append(vulnerability)
    return non_patchable_list


def extract_non_patchable_vulnerabilities(
    vulnerability_list_object: ECREnhancedScanVulnerabilityList, image_uri: str
):
    """
    This method takes a vulnerability_list_object for an image_uri and finds all the non-patchable packages in it. vulnerability_list_object consists of
    the vulnerabilities found in the latest scan for the image. It uses this object to see if there is any impacted package. It then invokes the methods
    that help determine if the packages are not patchable anymore. Based on this, it return a ECREnhancedScanVulnerabilityList object that only has
    non-patchable vulnerabilities with appropriate reasons in it.

    :param vulnerability_list_object: Object of type ECREnhancedScanVulnerabilityList, it consists of the vulns found in the latest scan
    :param image_uri: str, URI of the image
    :return: Object of type ECREnhancedScanVulnerabilityList, object that only non-patchable vulnerabilities with appropriate reasons in it.
    """
    assert vulnerability_list_object, "`vulnerability_list_object` cannot be None."
    segregated_package_names = segregate_impacted_package_names_based_on_manager(
        vulnerability_list_object
    )
    impacted_os_packages = segregated_package_names["os_packages"]

    docker_run_cmd = f"docker run -v {test_utils.get_repository_local_path()}:/deep-learning-containers  -id --entrypoint='/bin/bash' {image_uri} "
    container_id = run(f"{docker_run_cmd}").stdout.strip()
    docker_exec_cmd = f"docker exec -i {container_id}"
    container_setup_cmd = "apt-get update"
    run(f"{docker_exec_cmd} {container_setup_cmd}", hide=True)

    # Using the latest impact packages, we re-run miscellaneous_scripts/extract_apt_patch_data.py to see if there is any latest package that
    # can still be patched.
    new_apt_patch_evaluation_data = run_patch_evaluation_script_to_reevaluate_package_status(
        docker_exec_cmd=docker_exec_cmd,
        image_uri=image_uri,
        impacted_os_packages=impacted_os_packages,
    )
    # We then extract the patch evaluation data that was embedded in the DLC at the time of build.
    embedded_apt_patch_evaluation_data = {}
    display_embdedded_patch_eval_data_cmd = "cat /opt/aws/dlc/patch-details/os_summary.json"
    display_output = run(f"{docker_exec_cmd} {display_embdedded_patch_eval_data_cmd}", warn=True)
    if display_output.ok:
        embedded_apt_patch_evaluation_data = json.loads(display_output.stdout.strip())

    installed_python_package_version_dict = get_installed_python_packages_with_version(
        docker_exec_command=docker_exec_cmd
    )

    non_patchable_vulnerabilities_with_reason = copy.deepcopy(vulnerability_list_object)
    print(non_patchable_vulnerabilities_with_reason.vulnerability_list)
    patchable_packages = []
    for (
        package_name,
        vulnerabilities,
    ) in non_patchable_vulnerabilities_with_reason.vulnerability_list.items():
        package_manager = vulnerabilities[0].package_details.package_manager
        if package_manager not in ["OS", "PYTHONPKG"]:
            patchable_packages.append(package_name)
            continue
        if package_manager == "OS":
            # Using the new and the embedded patch evaluation data, we decipher if the package is upgradable anymore or not.
            is_package_upgradable, ignore_msg = get_os_package_upgradable_status(
                package_name, new_apt_patch_evaluation_data, embedded_apt_patch_evaluation_data
            )
            if is_package_upgradable:
                # If it is upgradable, we remove it from non_patchable_vulnerabilities_with_reason Object since it can be patched
                patchable_packages.append(package_name)
                continue
            for package_vulnerability in vulnerabilities:
                package_vulnerability.reason_to_ignore = ignore_msg
        elif package_manager == "PYTHONPKG":
            # Similary, for python packages, we filter the vulnerabilities that are allowlistable i.e. non-patchable and let the non-patchable
            # ones remain in the non_patchable_vulnerabilities_with_reason Object.
            allowlistable_python_vulns = get_non_patchable_python_vulnerabilities(
                vulnerability_list=vulnerabilities,
                installed_python_package_version_dict=installed_python_package_version_dict,
                docker_exec_command=docker_exec_cmd,
            )
            if allowlistable_python_vulns:
                non_patchable_vulnerabilities_with_reason.vulnerability_list[
                    package_name
                ] = allowlistable_python_vulns
            else:
                patchable_packages.append(package_name)

    # In the end, any patchable package is removed from the non_patchable_vulnerabilities_with_reason object.
    non_patchable_vulnerabilities_with_reason.vulnerability_list = {
        k: v
        for k, v in non_patchable_vulnerabilities_with_reason.vulnerability_list.items()
        if k not in patchable_packages
    }
    return non_patchable_vulnerabilities_with_reason
