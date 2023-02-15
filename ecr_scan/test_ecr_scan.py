import json
import os
from time import sleep
import boto3
import traceback
from invoke import run, Context
import sys
sys.path.insert(1, '/home/ubuntu/deep-learning-containers/ecr_scans')
from test_utils import *
from security import *
from packaging.version import Version
from packaging.specifiers import SpecifierSet
from config import is_ecr_scan_allowlist_feature_enabled

ALLOWLIST_FEATURE_ENABLED_IMAGES = {"mxnet": SpecifierSet(">=1.8.0,<1.9.0")}


def is_image_covered_by_allowlist_feature(image):
    """
    This method checks if the allowlist feature has been enabled for the image

    :param image: str, Image URI
    """
    image_framework, image_version = get_framework_and_version_from_tag(image)
    if image_framework not in ALLOWLIST_FEATURE_ENABLED_IMAGES or any(substring in image for substring in ["example"]):
        return False
    if Version(image_version) in ALLOWLIST_FEATURE_ENABLED_IMAGES[image_framework]:
        return True
    return False


def is_generic_image():
    return os.getenv("IS_GENERIC_IMAGE", "false").lower() == "true"

def get_minimum_sev_threshold_level(image):
    """
    This method gets the value for minimum threshold level. This threshold level determines the
    vulnerability severity above which we want to raise an alarm. 

    :param image: str Image URI for which threshold has to be set
    """
    if is_generic_image():
        return "HIGH"
    if is_image_covered_by_allowlist_feature(image):
        return "MEDIUM"
    return "HIGH"

def get_target_image_uri_using_current_uri_and_target_repo(image, target_repository_name, target_repository_region, append_tag=""):
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
    registry = get_ecr_registry(account_id, target_repository_region)
    original_image_repository, original_image_tag = get_repository_and_tag_from_image_uri(image)
    if append_tag:
        upgraded_image_tag = f"{original_image_repository}-{original_image_tag}-{append_tag}"
    else:
        upgraded_image_tag = f"{original_image_repository}-{original_image_tag}"
    target_image_uri = f"{registry}/{target_repository_name}:{upgraded_image_tag}"
    return target_image_uri

def format_and_generate_report(ecr_image_vulnerability_list,image_scan_allowlist):
    ecr_report = json.loads(ecr_image_vulnerability_list)
    allowlist_report = json.loads(image_scan_allowlist)
    vulnerability_dict = {}
    for package in ecr_report:
        vulnerability_details = []
        scan_status = "N/A"
        if package in allowlist_report:
            print("Package exists in allowlist")
            print(package)
            print(allowlist_report[package])
            scan_status = "IGNORED"
            for cve in ecr_report[package]:
                vulnerability_details.append({
                    "vulnerability_id": cve["vulnerability_id"],
                    "description": cve["description"],
                    "source": cve["source"],
                    "severity": cve["severity"]
                })
        else:
            scan_status = "FAILED"
            for cve in ecr_report[package]:
                vulnerability_details.append({
                    "vulnerability_id": cve["vulnerability_id"],
                    "description": cve["description"],
                    "source": cve["source"],
                    "severity": cve["severity"]
                })
        vulnerability_dict[package] = {
                "scan_status" : scan_status,
                "installed": ecr_report[package][0]["package_details"]["version"],
                "vulnerabilities": vulnerability_details
            }
    print(vulnerability_dict)
    return vulnerability_dict

def test_ecr_enhanced_scan(image, ecr_client, sts_client, region):
    """
    Run ECR Enhanced Scan Tool on an image being tested, and raise Error if vulnerabilities found
    1. Upload image to the ECR Enhanced Scanning Testing Repo.
    2. Wait for the scans to complete - takes approx 10 minutes for big images. Once the scan is complete, 
        the scan status changes to ACTIVE
    3. If the status does not turn to ACTIVE, raise a TimeOut Error
    4. Read the ecr_scan_results and remove the allowlisted vulnerabilities from it
    5. In case any vulnerability is remaining after removal, raise an error

    :param image: str Image URI for image to be tested
    :param ecr_client: boto3 Client for ECR
    :param sts_client: boto3 Client for STS
    :param region: str Name of region where test is executed
    """
    LOGGER.info(f"Running test_ecr_enhanced_scan for image {image}")
    ecr_enhanced_repo_uri = get_target_image_uri_using_current_uri_and_target_repo(
        image,
        target_repository_name=ECR_ENHANCED_SCANNING_REPO_NAME,
        target_repository_region=ECR_ENHANCED_REPO_REGION,
        append_tag="ENHSCAN",
    )


    run(f"docker tag {image} {ecr_enhanced_repo_uri}", hide=True)
    reupload_image_to_test_ecr(
        ecr_enhanced_repo_uri, ECR_ENHANCED_SCANNING_REPO_NAME, ECR_ENHANCED_REPO_REGION, pull_image=False
    )

    ecr_client_for_enhanced_scanning_repo = boto3.client("ecr", region_name=ECR_ENHANCED_REPO_REGION)
    wait_for_enhanced_scans_to_complete(ecr_client_for_enhanced_scanning_repo, ecr_enhanced_repo_uri)
    sleep(1 * 60)

    scan_results = get_all_ecr_enhanced_scan_findings(
        ecr_client=ecr_client_for_enhanced_scanning_repo, image_uri=ecr_enhanced_repo_uri
    )
    with open('data.json', 'w') as f:
        json.dump(json.dumps(scan_results, cls=EnhancedJSONEncoder), f)
    scan_results = json.loads(json.dumps(scan_results, cls=EnhancedJSONEncoder))
    
    minimum_sev_threshold = get_minimum_sev_threshold_level(image)
    print("minimum_sev_threshold",minimum_sev_threshold)
    ecr_image_vulnerability_list = ECREnhancedScanVulnerabilityList(minimum_severity=CVESeverity[minimum_sev_threshold])
    ecr_image_vulnerability_list.construct_allowlist_from_ecr_scan_result(scan_results)

    LOGGER.info(f"ecr_image_vulnerability_list formed {ecr_image_vulnerability_list.vulnerability_list}")

    image_scan_allowlist = ECREnhancedScanVulnerabilityList(minimum_severity=CVESeverity[minimum_sev_threshold])
    try:
        # Derive Image Scan Allowlist Path
        if is_generic_image():
            image_scan_allowlist_path = get_allowlist_path_for_enhanced_scan_from_env_variable()
        else:
            image_scan_allowlist_path = get_ecr_scan_allowlist_path(image)
        LOGGER.info(f"[Allowlist] Trying to locate Allowlist at PATH: {image_scan_allowlist_path}")
        # Check if image Scan Allowlist Path exists
        if os.path.exists(image_scan_allowlist_path):
            image_scan_allowlist.construct_allowlist_from_file(image_scan_allowlist_path)
            LOGGER.info(f"[Allowlist] Using allowlist at location {image_scan_allowlist_path} to skip {image_scan_allowlist.get_summarized_info()}")
    except:
        LOGGER.info(f"[Allowlist] Image scan allowlist path could not be derived for {image}")
        traceback.print_exc()

    ecr_list = json.dumps(ecr_image_vulnerability_list.vulnerability_list, cls= EnhancedJSONEncoder)
    allow_list = json.dumps(image_scan_allowlist.vulnerability_list, cls= EnhancedJSONEncoder)
    formated_report = format_and_generate_report(ecr_list,allow_list)
    
    LOGGER.info(f"ECR Enhanced Scanning test completed for image: {image}")
    vulnerability_list_to_embed = json.dumps(formated_report, cls= EnhancedJSONEncoder)
    print(vulnerability_list_to_embed)
    return vulnerability_list_to_embed
    # if remaining_vulnerabilities:
    #     assert not remaining_vulnerabilities.vulnerability_list, (
    #         f"Total of {len(remaining_vulnerabilities.vulnerability_list)} vulnerabilities need to be fixed on {image}:\n"
    #         f"{json.dumps(remaining_vulnerabilities.vulnerability_list, cls= EnhancedJSONEncoder)}"
    #     )
        



