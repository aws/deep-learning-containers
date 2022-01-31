import json
import os

from time import sleep, time

import pytest
import boto3

from copy import deepcopy
from invoke import run, Context

from test.test_utils import LOGGER, get_account_id_from_image_uri, get_dockerfile_path_for_image, is_dlc_cicd_context
from test.test_utils import ecr as ecr_utils
from test.test_utils.security import (
    CVESeverity, ScanVulnerabilityList, ECRScanFailureException, get_ecr_scan_allowlist_path
)
from src.config import is_ecr_scan_allowlist_feature_enabled


MINIMUM_SEV_THRESHOLD = "HIGH"

LOWER_THRESHOLD_IMAGES = {"mxnet":["1.8"]}

@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("N/A")
@pytest.mark.canary("Run security test regularly on production images")
def test_security(image):
    repo_name, image_tag = image.split("/")[-1].split(":")
    container_name = f"{repo_name}-{image_tag}-security"

    run(
        f"docker run -itd --name {container_name} "
        f"--mount type=bind,src=$(pwd)/container_tests,target=/test"
        f" --entrypoint='/bin/bash' "
        f"{image}",
        echo=True,
    )
    try:
        docker_exec_cmd = f"docker exec -i {container_name}"
        run(f"{docker_exec_cmd} python /test/bin/security_checks.py ")
    finally:
        run(f"docker rm -f {container_name}", hide=True)


def run_scan(ecr_client, image):
    scan_status = None
    start_time = time()
    ecr_utils.start_ecr_image_scan(ecr_client, image)
    while (time() - start_time) <= 600:
        scan_status, scan_status_description = ecr_utils.get_ecr_image_scan_status(ecr_client, image)
        if scan_status == "FAILED" or scan_status not in [None, "IN_PROGRESS", "COMPLETE"]:
            raise ECRScanFailureException(f"ECR Scan failed for {image} with description: {scan_status_description}")
        if scan_status == "COMPLETE":
            break
        sleep(1)
    if scan_status != "COMPLETE":
        raise TimeoutError(f"ECR Scan is still in {scan_status} state. Exiting.")

def get_new_image_uri(image):
    """
    Returns the new image uri for the image being tested. After running the apt commands, the
    new image will be uploaded to the ECR based on the new image uri.

    :param image: str
    :param new_image_uri: str
    """
    repository_name = os.getenv('UPGRADE_REPO_NAME')
    ecr_account = f"{os.getenv('ACCOUNT_ID')}.dkr.ecr.{os.getenv('REGION')}.amazonaws.com"
    upgraded_image_tag = '-'.join(image.replace("/",":").split(":")[1:]) + "-up"
    new_image_uri = f"{ecr_account}/{repository_name}:{upgraded_image_tag}"
    return new_image_uri

def run_upgrade_on_image_and_push(image, new_image_uri):
    """
    Creates a container for the image being tested. Runs apt update and upgrade on the container
    and the commits the container as new_image_uri. This new image is then pushed to the ECR. 

    :param image: str
    :param new_image_uri: str
    """
    ctx = Context()
    docker_run_cmd = f"docker run -id --entrypoint='/bin/bash' {image}"
    container_id = ctx.run(f"{docker_run_cmd}", hide=True, warn=True).stdout.strip()
    apt_command = "apt-get update && apt-get upgrade"
    docker_exec_cmd = f"docker exec -i {container_id}"
    run_output = ctx.run(f"{docker_exec_cmd} {apt_command}", hide=True, warn=True)
    if not run_output.ok:
        raise ValueError(f"Could not run apt update and upgrade. \n" \
                         f"Stdout is {run_output.stdout} \n" \
                         f"Stderr is {run_output.stderr} \n" \
                         f"Failed status is {run_output.exited}")
    ctx.run(f"docker commit {container_id} {new_image_uri}", hide=True, warn=True)
    ctx.run(f"docker rm -f {container_id}", hide=True, warn=True)
    ctx.run(f"docker push {new_image_uri}", hide=True, warn=True)

def invoke_lambda(function_name, payload_dict = {}):
    """
    Asyncronously Invokes the passed lambda.

    :param function_name: str, name of the lambda function
    :param payload_dict: dict, payload to be sent to the lambda
    """
    lambda_client = boto3.client('lambda', region_name=os.getenv('REGION'))
    response = lambda_client.invoke(
        FunctionName=function_name,
        InvocationType='Event',
        LogType='Tail',
        Payload=json.dumps(payload_dict)
    )
    status_code = response.get('StatusCode')
    if status_code != 202:
        raise ValueError("Lambda call not made properly. Status code returned {status_code}")

def create_and_save_package_list_to_s3(old_filepath, new_packages, new_filepath):
    """
    This method conducts the union of packages present in the original apt-get-upgrade
    list and new list of packages passed as an argument. It makes a new file and stores
    the results in it.
    :param old_filpath: str, path of original file
    :param new_packages: list[str], consists of list of packages
    :param new_filpath: str, path of new file that will have the results of union
    """
    file1 = open(old_filepath,"r")
    lines = file1.readlines()
    current_packages = [line.strip() for line in lines]
    package_list = current_packages + new_packages
    package_list = [f"{package_name}\n" for package_name in package_list]
    file1.close()
    run(f"rm -rf {new_filepath}")
    file2 = open(new_filepath,"w")
    file2.writelines(package_list)
    file2.close()
    s3_bucket_name = "trshanta-bucket"
    s3_client = boto3.client("s3")
    s3_client.upload_file(Filename=new_filepath, Bucket=s3_bucket_name, Key=new_filepath)

def save_allowlist_to_s3(image, vulnerability_list):
    """
    Saves the vulnerability list in the s3 bucket. It uses image to decide the name of the file on 
    the s3 bucket.

    :param image: str, image uri 
    :param vulnerability_list: ScanVulnerabilityList
    :return: str, name of the file as stored on s3
    """
    s3_bucket_name = "trshanta-bucket"
    processed_image_uri = image.replace(".", "-").replace("/", "-").replace(":", "-")
    file_name = f"{processed_image_uri}-allowlist.json"
    vulnerability_list.save_vulnerability_list(file_name)
    s3_client = boto3.client("s3")
    s3_client.upload_file(Filename=file_name, Bucket=s3_bucket_name, Key=file_name)
    return file_name

def get_vulnerabilites_fixable_by_upgrade(image_allowlist, ecr_image_vulnerability_list, upgraded_image_vulnerability_list):
    """
    Finds out the vulnerabilities that are fixable by apt-get update and apt-get upgrade.

    :param image_allowlist: ScanVulnerabilityList, Vulnerabities that are present in the respective allowlist in the 
                            DLC git repo.
    :param ecr_image_vulnerability_list: ScanVulnerabilityList, Vulnerabities recently detected WITHOUT running apt-upgrade on 
                                         the originally released image.
    :param upgraded_image_vulnerability_list: ScanVulnerabilityList, Vulnerabilites exisiting in the image WITH apt-upgrade 
                                              run on it.
    :return: ScanVulnerabilityList/NONE, either ScanVulnerabilityList object or None if no fixable vulnerability
    """
    fixable_ecr_image_scan_vulnerabilites = ecr_image_vulnerability_list - upgraded_image_vulnerability_list
    fixable_allowlist_vulnerabilites = image_allowlist - upgraded_image_vulnerability_list
    vulnerabilities_fixable_by_upgrade = None
    if fixable_ecr_image_scan_vulnerabilites and fixable_allowlist_vulnerabilites:
        vulnerabilities_fixable_by_upgrade = fixable_ecr_image_scan_vulnerabilites + fixable_allowlist_vulnerabilites
    elif fixable_ecr_image_scan_vulnerabilites:
        vulnerabilities_fixable_by_upgrade = fixable_ecr_image_scan_vulnerabilites
    elif fixable_allowlist_vulnerabilites:
        vulnerabilities_fixable_by_upgrade = fixable_allowlist_vulnerabilites
    return vulnerabilities_fixable_by_upgrade


def conduct_failure_routine(image, image_allowlist, ecr_image_vulnerability_list, upgraded_image_vulnerability_list):
    s3_filename_for_allowlist = save_allowlist_to_s3(image, upgraded_image_vulnerability_list)
    original_filepath_for_allowlist = get_ecr_scan_allowlist_path(image)
    edited_files = [{"s3_filename": s3_filename_for_allowlist, "github_filepath": original_filepath_for_allowlist}]
    vulnerabilities_fixable_by_upgrade = get_vulnerabilites_fixable_by_upgrade(image_allowlist, ecr_image_vulnerability_list, upgraded_image_vulnerability_list)
    newly_found_non_fixable_vulnerabilites = upgraded_image_vulnerability_list - image_allowlist
    fixable_list = []
    if vulnerabilities_fixable_by_upgrade:
        fixable_list = vulnerabilities_fixable_by_upgrade.vulnerability_list
    s3_filename_for_apt_upgrade_list = s3_filename_for_allowlist.replace("allowlist.json","apt-upgrade-list.txt")
    original_filepath_for_apt_upgrade_list = '/'.join(original_filepath_for_allowlist.split('/')[:-1] + ["apt-upgrade-list.txt"])
    new_package_list = fixable_list if isinstance(fixable_list, list) else list(fixable_list.keys())
    create_and_save_package_list_to_s3(original_filepath_for_apt_upgrade_list, new_package_list, s3_filename_for_apt_upgrade_list)
    edited_files.append({"s3_filename": s3_filename_for_apt_upgrade_list, "github_filepath": original_filepath_for_apt_upgrade_list})
    newly_found_non_fixable_list = []
    if newly_found_non_fixable_vulnerabilites:
        newly_found_non_fixable_list = newly_found_non_fixable_vulnerabilites.vulnerability_list
    message_body = {
        "edited_files": edited_files,
        "fixable_vulnerabilities": fixable_list,
        "non_fixable_vulnerabilities": newly_found_non_fixable_list
    }
    invoke_lambda(function_name = 'trshanta-ECR-AS', payload_dict=message_body)
    return_dict = deepcopy(message_body)
    return_dict["s3_filename_for_allowlist"] = s3_filename_for_allowlist
    return return_dict

def is_image_covered_by_allowlist_feature(image):
    """
    This method checks if the allowlist feature has been enabled for the image

    :param image: str, Image URI
    """
    for framework in LOWER_THRESHOLD_IMAGES.keys():
        if framework in image:
            if any(version in image for version in LOWER_THRESHOLD_IMAGES[framework]):
                return True
            return False
    return False

def set_minimum_threshold_level(image):
    """
    This method sets the value for MINIMUM_SEV_THRESHOLD

    :param image: str Image URI for which threshold has to be set
    """
    global MINIMUM_SEV_THRESHOLD
    if is_image_covered_by_allowlist_feature(image):
        MINIMUM_SEV_THRESHOLD = "MEDIUM"
        return
    MINIMUM_SEV_THRESHOLD = "HIGH"

def fetch_other_vulnerability_lists(image, ecr_client):
    new_image_uri = get_new_image_uri(image)
    run_upgrade_on_image_and_push(image, new_image_uri)
    run_scan(ecr_client, new_image_uri)
    scan_results_with_upgrade = ecr_utils.get_ecr_image_scan_results(ecr_client, new_image_uri, minimum_vulnerability=MINIMUM_SEV_THRESHOLD)
    scan_results_with_upgrade = ecr_utils.populate_ecr_scan_with_web_scraper_results(new_image_uri, scan_results_with_upgrade)
    upgraded_image_vulnerability_list = ScanVulnerabilityList(minimum_severity=CVESeverity[MINIMUM_SEV_THRESHOLD])
    upgraded_image_vulnerability_list.construct_allowlist_from_ecr_scan_result(scan_results_with_upgrade)
    image_scan_allowlist = ScanVulnerabilityList(minimum_severity=CVESeverity[MINIMUM_SEV_THRESHOLD])
    image_scan_allowlist_path = get_ecr_scan_allowlist_path(image)
    if os.path.exists(image_scan_allowlist_path):
        image_scan_allowlist.construct_allowlist_from_file(image_scan_allowlist_path)
    return upgraded_image_vulnerability_list, image_scan_allowlist


def log_all_lists_in_s3(failure_routine_summary):
    s3_filename_for_allowlist = failure_routine_summary["s3_filename_for_allowlist"]
    s3_filename_for_fixable_list = s3_filename_for_allowlist.replace("allowlist.json","fixable-vulnerability-list.json")
    s3_filename_for_non_fixable_list = s3_filename_for_allowlist.replace("allowlist.json","non-fixable-vulnerability-list.json")
    s3_bucket_name = "trshanta-bucket"
    s3_client = boto3.client("s3")

    with open(s3_filename_for_fixable_list, "w") as outfile:
        json.dump(failure_routine_summary["fixable_vulnerabilities"], outfile, indent=4)
    s3_client.upload_file(Filename=s3_filename_for_fixable_list, Bucket=s3_bucket_name, Key=s3_filename_for_fixable_list)

    with open(s3_filename_for_non_fixable_list, "w") as outfile:
        json.dump(failure_routine_summary["non_fixable_vulnerabilities"], outfile, indent=4)
    s3_client.upload_file(Filename=s3_filename_for_non_fixable_list, Bucket=s3_bucket_name, Key=s3_filename_for_non_fixable_list)
    return s3_filename_for_fixable_list, s3_filename_for_non_fixable_list

@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("N/A")
@pytest.mark.canary("Run ECR Scan test regularly on production images")
@pytest.mark.integration("check OS dependencies")
def test_ecr_scan(image, ecr_client, sts_client, region):
    """
    Run ECR Scan Tool on an image being tested, and raise Error if vulnerabilities found
    1. Start Scan.
    2. For 5 minutes (Run DescribeImages):
       (We run this for 5 minutes because the Scan is expected to complete in about 2 minutes, though no
        analysis has been performed on exactly how long the Scan takes for a DLC image. Therefore we also
        have a 3 minute buffer beyond the expected amount of time taken.)
    3.1. If imageScanStatus == COMPLETE: exit loop
    3.2. If imageScanStatus == IN_PROGRESS or AttributeNotFound(imageScanStatus): continue loop
    3.3. If imageScanStatus == FAILED: raise RuntimeError
    4. If DescribeImages.imageScanStatus != COMPLETE: raise TimeOutError
    5. assert imageScanFindingsSummary.findingSeverityCounts.HIGH/CRITICAL == 0

    :param image: str Image URI for image to be tested
    :param ecr_client: boto3 Client for ECR
    :param sts_client: boto3 Client for STS
    :param region: str Name of region where test is executed
    """
    test_account_id = sts_client.get_caller_identity().get("Account")
    image_account_id = get_account_id_from_image_uri(image)
    if image_account_id != test_account_id:
        image_repo_uri, image_tag = image.split(":")
        _, image_repo_name = image_repo_uri.split("/")
        target_image_repo_name = f"beta-{image_repo_name}"
        image = ecr_utils.reupload_image_to_test_ecr(image, target_image_repo_name, region)
    
    set_minimum_threshold_level(image)
    print(MINIMUM_SEV_THRESHOLD)

    run_scan(ecr_client, image)
    scan_results = ecr_utils.get_ecr_image_scan_results(ecr_client, image, minimum_vulnerability=MINIMUM_SEV_THRESHOLD)
    scan_results = ecr_utils.populate_ecr_scan_with_web_scraper_results(image, scan_results)
    ecr_image_vulnerability_list = ScanVulnerabilityList(minimum_severity=CVESeverity[MINIMUM_SEV_THRESHOLD])
    ecr_image_vulnerability_list.construct_allowlist_from_ecr_scan_result(scan_results)

    remaining_vulnerabilities = ecr_image_vulnerability_list

    # TODO: Once this feature is enabled, remove "if" condition and second assertion statement
    # TODO: Ensure this works on the canary tags before removing feature flag
    if is_image_covered_by_allowlist_feature(image):
        upgraded_image_vulnerability_list, image_scan_allowlist = fetch_other_vulnerability_lists(image, ecr_client)

        ## In case new vulnerabilities are found conduct failure routine
        newly_found_vulnerabilities = ecr_image_vulnerability_list - image_scan_allowlist
        if newly_found_vulnerabilities:
            failure_routine_summary = conduct_failure_routine(image, image_scan_allowlist, ecr_image_vulnerability_list, upgraded_image_vulnerability_list)
            s3_filename_for_fixable_list, s3_filename_for_non_fixable_list = log_all_lists_in_s3(failure_routine_summary)
        
        assert not newly_found_vulnerabilities, (
            f"""Found {len(failure_routine_summary["fixable_vulnerabilities"])} fixable vulnerabilites """
            f"""and {len(failure_routine_summary["non_fixable_vulnerabilities"])} non fixable vulnerabilites. """
            f"""Refer to files {s3_filename_for_fixable_list}, {s3_filename_for_non_fixable_list} """
            f"""and {failure_routine_summary["s3_filename_for_allowlist"]} in trshanta-bucket."""
        )

        ## In case there is no new vulnerability but the allowlist is outdated conduct failure routine
        vulnerabilities_that_can_be_fixed = image_scan_allowlist - upgraded_image_vulnerability_list
        if vulnerabilities_that_can_be_fixed:
            conduct_failure_routine(image, image_scan_allowlist, ecr_image_vulnerability_list, upgraded_image_vulnerability_list)
        assert not vulnerabilities_that_can_be_fixed, (
            f"Allowlist is outdated. The following vulnerabilities can be fixed {image}:\n"
            f"{json.dumps(vulnerabilities_that_can_be_fixed.vulnerability_list, indent=4)}"
        )
        return

    assert not remaining_vulnerabilities.vulnerability_list, (
        f"The following vulnerabilities need to be fixed on {image}:\n"
        f"{json.dumps(remaining_vulnerabilities.vulnerability_list, indent=4)}"
    )


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("N/A")
@pytest.mark.integration("check OS dependencies")
@pytest.mark.skipif(is_dlc_cicd_context(), reason="Temporarily allow slack in allowlist w.r.t. actual vulnerabilities")
def test_is_ecr_scan_allowlist_outdated(image, ecr_client, sts_client, region):
    """
    Run ECR Scan Tool on an image being tested, and test if the vulnerabilities in the allowlist for the image
    are still valid, or if any vulnerabilities must be removed from the list.

    :param image: str Image URI for image to be tested
    :param ecr_client: boto3 Client for ECR
    :param sts_client: boto3 Client for STS
    :param region: str Name of region where test is executed
    """
    test_account_id = sts_client.get_caller_identity().get("Account")
    image_account_id = get_account_id_from_image_uri(image)
    if image_account_id != test_account_id:
        image_repo_uri, image_tag = image.split(":")
        _, image_repo_name = image_repo_uri.split("/")
        target_image_repo_name = f"beta-{image_repo_name}"
        image = ecr_utils.reupload_image_to_test_ecr(image, target_image_repo_name, region)

    run_scan(ecr_client, image)
    scan_results = ecr_utils.get_ecr_image_scan_results(ecr_client, image, minimum_vulnerability=MINIMUM_SEV_THRESHOLD)
    scan_results = ecr_utils.populate_ecr_scan_with_web_scraper_results(image, scan_results)

    image_scan_allowlist = ScanVulnerabilityList(minimum_severity=CVESeverity[MINIMUM_SEV_THRESHOLD])
    image_scan_allowlist_path = get_ecr_scan_allowlist_path(image)
    if os.path.exists(image_scan_allowlist_path):
        image_scan_allowlist.construct_allowlist_from_file(image_scan_allowlist_path)

    ecr_image_vulnerability_list = ScanVulnerabilityList(minimum_severity=CVESeverity[MINIMUM_SEV_THRESHOLD])
    ecr_image_vulnerability_list.construct_allowlist_from_ecr_scan_result(scan_results)

    invalid_allowlist_vulnerabilities = image_scan_allowlist - ecr_image_vulnerability_list

    assert not invalid_allowlist_vulnerabilities, (
        f"The following vulnerabilities are no longer valid on {image}:\n"
        f"{json.dumps(invalid_allowlist_vulnerabilities.vulnerability_list, indent=4)}"
    )
