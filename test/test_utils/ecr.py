import json
import os
import subprocess

from base64 import b64decode

import boto3
import botocore

from test.test_utils import (
    get_repository_and_tag_from_image_uri,
    get_region_from_image_uri,
    get_account_id_from_image_uri,
    get_unique_name_from_tag,
    get_repository_local_path,
    LOGGER,
)
from test.test_utils.security import CVESeverity


class ECRScanFailedError(Exception):
    pass


class ECRRepoDoesNotExist(Exception):
    pass


def _botocore_resolver():
    """
    Get the DNS suffix for the given region.
    :return: endpoint object
    """
    loader = botocore.loaders.create_loader()
    return botocore.regions.EndpointResolver(loader.load_data("endpoints"))


def get_ecr_registry(account, region):
    """
    Get prefix of ECR image URI
    :param account: Account ID
    :param region: region where ECR repo exists
    :return: AWS ECR registry
    """
    endpoint_data = _botocore_resolver().construct_endpoint("ecr", region)
    return "{}.dkr.{}".format(account, endpoint_data["hostname"])


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
        scan_info = ecr_client.describe_image_scan_findings(
            repositoryName=repository, imageId={"imageTag": tag}
        )
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
        scan_info = ecr_client.start_image_scan(
            repositoryName=repository, imageId={"imageTag": tag}
        )
    except ecr_client.exceptions.LimitExceededException:
        LOGGER.warning("Scan has already been run on this image in the last 24 hours.")
        return
    if scan_info["imageScanStatus"]["status"] == "FAILED":
        raise ECRScanFailedError(
            f"ECR Scan failed and returned:\n{json.dumps(scan_info, indent=4)}"
        )
    return


def get_ecr_image_scan_status(ecr_client, image_uri):
    """
    Get status of an ECR image scan in progress
    :param ecr_client: boto3 client for ECR
    :param image_uri: image URI for image to be checked
    :return: tuple<str, str> Scan Status, Status Description
    """
    repository, tag = get_repository_and_tag_from_image_uri(image_uri)
    image_info = ecr_client.describe_images(
        repositoryName=repository, imageIds=[{"imageTag": tag}]
    )["imageDetails"][0]
    if "imageScanStatus" not in image_info.keys():
        return None, "Scan not started"
    return image_info["imageScanStatus"]["status"], image_info["imageScanStatus"].get(
        "description", "NO DESCRIPTION"
    )


def get_ecr_image_enhanced_scan_status(ecr_client, image_uri):
    """
    Get status of an ECR Enhanced image scan.
    :param ecr_client: boto3 client for ECR
    :param image_uri: image URI for image to be checked
    :return: tuple<str, str> Scan Status, Status Description
    """
    repository, tag = get_repository_and_tag_from_image_uri(image_uri)
    scan_info = ecr_client.describe_image_scan_findings(
        repositoryName=repository, imageId={"imageTag": tag}, maxResults=1
    )
    return scan_info["imageScanStatus"]["status"], scan_info["imageScanStatus"]["description"]


def get_ecr_image_scan_severity_count(ecr_client, image_uri):
    """
    Get ECR image scan findings
    :param ecr_client: boto3 client for ECR
    :param image_uri: image URI for image to be checked
    :return: dict ECR Scan Findings
    """
    repository, tag = get_repository_and_tag_from_image_uri(image_uri)
    scan_info = ecr_client.describe_image_scan_findings(
        repositoryName=repository, imageId={"imageTag": tag}
    )
    severity_counts = scan_info["imageScanFindings"]["findingSeverityCounts"]
    return severity_counts


def get_all_ecr_image_scan_results(ecr_client, image_uri, scan_info_finding_key="enhancedFindings"):
    """
    Get list of All vulnerabilities from ECR image scan results using pagination
    :param ecr_client: boto3 ecr client
    :param image_uri: str, image uri
    :return: list<dict> Scan results
    """
    scan_info_findings = []
    registry_id = get_account_id_from_image_uri(image_uri)
    repository, tag = get_repository_and_tag_from_image_uri(image_uri)
    paginator = ecr_client.get_paginator("describe_image_scan_findings")
    response_iterator = paginator.paginate(
        registryId=registry_id,
        repositoryName=repository,
        imageId={"imageTag": tag},
        PaginationConfig={
            "PageSize": 50,
        },
    )
    for page in response_iterator:
        if scan_info_finding_key in page["imageScanFindings"]:
            scan_info_findings += page["imageScanFindings"][scan_info_finding_key]
    LOGGER.info(
        f"[TotalVulnsFound] For image_uri: {image_uri} {len(scan_info_findings)} vulnerabilities found in total."
    )
    return scan_info_findings


def get_ecr_image_scan_results(ecr_client, image_uri, minimum_vulnerability="HIGH"):
    """
    Get list of vulnerabilities from ECR image scan results
    :param ecr_client:
    :param image_uri:
    :param minimum_vulnerability: str representing minimum vulnerability level to report in results
    :return: list<dict> Scan results
    """
    repository, tag = get_repository_and_tag_from_image_uri(image_uri)
    scan_info = ecr_client.describe_image_scan_findings(
        repositoryName=repository, imageId={"imageTag": tag}
    )
    scan_findings = [
        finding
        for finding in scan_info["imageScanFindings"]["findings"]
        if CVESeverity[finding["severity"]] >= CVESeverity[minimum_vulnerability]
    ]
    return scan_findings


def get_all_ecr_enhanced_scan_findings(ecr_client, image_uri):
    """
    Get list of all vulnerabilities from ECR ENHANCED image scan results
    :param ecr_client:
    :param image_uri:
    :return: list<dict> Scan results
    """
    scan_info_findings = get_all_ecr_image_scan_results(
        ecr_client, image_uri, scan_info_finding_key="enhancedFindings"
    )
    return scan_info_findings


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
    for auth in result["authorizationData"]:
        auth_token = b64decode(auth["authorizationToken"]).decode()
        user_name, password = auth_token.split(":")
    return user_name, password


def save_credentials_to_file(file_path, password):
    with open(file_path, "w") as file:
        file.write(f"{password}")


def delete_file(file_path):
    subprocess.check_output(f"rm -rf {file_path}", shell=True, executable="/bin/bash")


def reupload_image_to_test_ecr(
    source_image_uri, target_image_repo_name, target_region, pull_image=True
):
    """
    Helper function to reupload an image owned by a another/same account to an ECR repo in this account to given region, so that
    this account can freely run tests without permission issues.

    :param source_image_uri: str Image URI for image to be tested
    :param target_image_repo_name: str Target image ECR repo name
    :param target_region: str Region where test is being run
    :param pull_image: bool, specifies if the source_image needs to be pulled before reuploading
    :return: str New image URI for re-uploaded image
    """
    ECR_PASSWORD_FILE_PATH = os.path.join(
        "/tmp", f"{get_unique_name_from_tag(source_image_uri)}.txt"
    )
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

    client = boto3.client("ecr", region_name=image_region)
    username, password = get_ecr_login_boto3(client, image_account_id, image_region)
    save_credentials_to_file(ECR_PASSWORD_FILE_PATH, password)

    # using ctx.run throws error on codebuild "OSError: reading from stdin while output is captured".
    # Also it throws more errors related to awscli if in_stream=False flag is added to ctx.run which needs more deep dive
    if pull_image:
        LOGGER.info(f"reupload_image_to_test_ecr: pulling {source_image_uri}")
        subprocess.check_output(
            f"cat {ECR_PASSWORD_FILE_PATH} | docker login -u {username} --password-stdin https://{image_account_id}.dkr.ecr.{image_region}.amazonaws.com && docker pull {source_image_uri}",
            shell=True,
            executable="/bin/bash",
        )
        LOGGER.info(f"reupload_image_to_test_ecr: pulling {source_image_uri} completed")
    subprocess.check_output(
        f"docker tag {source_image_uri} {target_image_uri}", shell=True, executable="/bin/bash"
    )
    delete_file(ECR_PASSWORD_FILE_PATH)
    username, password = get_ecr_login_boto3(target_ecr_client, target_account_id, target_region)
    save_credentials_to_file(ECR_PASSWORD_FILE_PATH, password)

    LOGGER.info(f"reupload_image_to_test_ecr: pushing {source_image_uri}")
    subprocess.check_output(
        f"cat {ECR_PASSWORD_FILE_PATH} | docker login -u {username} --password-stdin https://{target_account_id}.dkr.ecr.{target_region}.amazonaws.com && docker push {target_image_uri}",
        shell=True,
        executable="/bin/bash",
    )
    LOGGER.info(f"reupload_image_to_test_ecr: pushing {source_image_uri} completed")
    delete_file(ECR_PASSWORD_FILE_PATH)

    return target_image_uri
