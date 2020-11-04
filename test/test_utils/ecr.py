import json

from test.test_utils import get_repository_and_tag_from_image_uri, LOGGER


class ECRScanFailedError(Exception):
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
