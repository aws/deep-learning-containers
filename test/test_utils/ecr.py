import json

from enum import IntEnum

from test.test_utils import (
    get_repository_and_tag_from_image_uri,
    get_region_from_image_uri,
    get_account_id_from_image_uri,
    login_to_ecr_registry,
    get_unique_name_from_tag,
    LOGGER,
)
import subprocess
import os
import boto3
from base64 import b64decode
import sys


class CVESeverity(IntEnum):
    UNDEFINED = 0
    INFORMATIONAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


class ECRScanFailedError(Exception):
    pass


class ECRRepoDoesNotExist(Exception):
    pass


def get_ecr_image_scan_time(ecr_client, image_uri):
    """
    Find timestamp of when ECR Scan was last run for a given image URI assuming that this repository makes this
    information available to the account running this function.
    :param ecr_client: boto3 client for ECR
    :param image_uri: image URI for image to be checked
    :return: datetime.datetime object with UTC time or None
    """
    repository, tag = get_repository_and_tag_from_image_uri(image_uri)
    try:
        scan_info = ecr_client.describe_image_scan_findings(repositoryName=repository, imageId={"imageTag": tag})
    except ecr_client.exceptions.ScanNotFoundException:
        return None
    return scan_info["imageScanFindings"]["imageScanCompletedAt"]


def start_ecr_image_scan(ecr_client, image_uri):
    """
    Start ECR Scan for an image, and Warn if scan cannot be started
    :param ecr_client: boto3 client for ECR
    :param image_uri: image URI for image to be checked
    """
    repository, tag = get_repository_and_tag_from_image_uri(image_uri)
    try:
        scan_info = ecr_client.start_image_scan(repositoryName=repository, imageId={"imageTag": tag})
    except ecr_client.exceptions.LimitExceededException:
        LOGGER.warning("Scan has already been run on this image in the last 24 hours.")
        return
    if scan_info["imageScanStatus"]["status"] == "FAILED":
        raise ECRScanFailedError(f"ECR Scan failed and returned:\n{json.dumps(scan_info, indent=4)}")
    return


def get_ecr_image_scan_status(ecr_client, image_uri):
    """
    Get status of an ECR image scan in progress
    :param ecr_client: boto3 client for ECR
    :param image_uri: image URI for image to be checked
    :return: tuple<str, str> Scan Status, Status Description
    """
    repository, tag = get_repository_and_tag_from_image_uri(image_uri)
    image_info = ecr_client.describe_images(repositoryName=repository, imageIds=[{"imageTag": tag}])["imageDetails"][0]
    if "imageScanStatus" not in image_info.keys():
        return None, "Scan not started"
    return image_info["imageScanStatus"]["status"], image_info["imageScanStatus"].get("description", "NO DESCRIPTION")


def get_ecr_image_scan_severity_count(ecr_client, image_uri):
    """
    Get ECR image scan findings
    :param ecr_client: boto3 client for ECR
    :param image_uri: image URI for image to be checked
    :return: dict ECR Scan Findings
    """
    repository, tag = get_repository_and_tag_from_image_uri(image_uri)
    scan_info = ecr_client.describe_image_scan_findings(repositoryName=repository, imageId={"imageTag": tag})
    severity_counts = scan_info["imageScanFindings"]["findingSeverityCounts"]
    return severity_counts


def get_ecr_image_scan_results(ecr_client, image_uri, minimum_vulnerability="HIGH"):
    """
    Get list of vulnerabilities from ECR image scan results
    :param ecr_client:
    :param image_uri:
    :param minimum_vulnerability: str representing minimum vulnerability level to report in results
    :return: list<dict> Scan results
    """
    repository, tag = get_repository_and_tag_from_image_uri(image_uri)
    scan_info = ecr_client.describe_image_scan_findings(repositoryName=repository, imageId={"imageTag": tag})
    scan_findings = [
        finding
        for finding in scan_info["imageScanFindings"]["findings"]
        if CVESeverity[finding["severity"]] >= CVESeverity[minimum_vulnerability]
    ]
    return scan_findings


def ecr_repo_exists(ecr_client, repo_name, account_id=None):
    """
    :param ecr_client: boto3.Client for ECR
    :param repo_name: str ECR Repository Name
    :param account_id: str Account ID where repo is expected to exist
    :return: bool True if repo exists, False if not
    """
    query = {"repositoryNames": [repo_name]}
    if account_id:
        query["registryId"] = account_id
    try:
        ecr_client.describe_repositories(**query)
    except ecr_client.exceptions.RepositoryNotFoundException as e:
        return False
    return True


def get_ecr_login_boto3(ecr_client, account_id, region):
    """
    Get ECR login using boto3
    """
    user_name, password = None, None
    result = ecr_client.get_authorization_token()
    for auth in result['authorizationData']:
        auth_token = b64decode(auth['authorizationToken']).decode()
        user_name, password = auth_token.split(':')
    return user_name, password


def save_credentials_to_file(file_path, password):
    with open(file_path, "w") as file:
        file.write(f"{password}")


def delete_file(file_path):
    subprocess.check_output(f"rm -rf {file_path}", shell=True, executable="/bin/bash")


def reupload_image_to_test_ecr(source_image_uri, target_image_repo_name, target_region):
    """
    Helper function to reupload an image owned by a another/same account to an ECR repo in this account to given region, so that
    this account can freely run tests without permission issues.

    :param source_image_uri: str Image URI for image to be tested
    :param target_image_repo_name: str Target image ECR repo name
    :param target_region: str Region where test is being run
    :return: str New image URI for re-uploaded image
    """
    ECR_PASSWORD_FILE_PATH = os.path.join("/tmp", f"{get_unique_name_from_tag(source_image_uri)}.txt")
    sts_client = boto3.client("sts", region_name=target_region)
    target_ecr_client = boto3.client("ecr", region_name=target_region)
    target_account_id = sts_client.get_caller_identity().get("Account")
    image_account_id = get_account_id_from_image_uri(source_image_uri)
    image_region = get_region_from_image_uri(source_image_uri)
    image_repo_uri, image_tag = source_image_uri.split(":")
    _, image_repo_name = image_repo_uri.split("/")
    if not ecr_repo_exists(target_ecr_client, target_image_repo_name):
        raise ECRRepoDoesNotExist(
            f"Repo named {target_image_repo_name} does not exist in {target_region} on the account {target_account_id}"
        )

    target_image_uri = (
        source_image_uri.replace(image_region, target_region)
        .replace(image_repo_name, target_image_repo_name)
        .replace(image_account_id, target_account_id)
    )

    client = boto3.client('ecr', region_name = image_region)
    username, password = get_ecr_login_boto3(client, image_account_id, image_region)
    save_credentials_to_file(ECR_PASSWORD_FILE_PATH, password)

    # using ctx.run throws error on codebuild "OSError: reading from stdin while output is captured".
    # Also it throws more errors related to awscli if in_stream=False flag is added to ctx.run which needs more deep dive
    subprocess.check_output(f"cat {ECR_PASSWORD_FILE_PATH} | docker login -u {username} --password-stdin https://{image_account_id}.dkr.ecr.{image_region}.amazonaws.com && docker pull {source_image_uri}", shell=True, executable="/bin/bash")
    subprocess.check_output(f"docker tag {source_image_uri} {target_image_uri}", shell=True, executable="/bin/bash")
    delete_file(ECR_PASSWORD_FILE_PATH)
    username, password = get_ecr_login_boto3(target_ecr_client, target_account_id, target_region)
    save_credentials_to_file(ECR_PASSWORD_FILE_PATH, password)
    subprocess.check_output(f"cat {ECR_PASSWORD_FILE_PATH} | docker login -u {username} --password-stdin https://{target_account_id}.dkr.ecr.{target_region}.amazonaws.com && docker push {target_image_uri}", shell=True, executable="/bin/bash")
    delete_file(ECR_PASSWORD_FILE_PATH)

    return target_image_uri
