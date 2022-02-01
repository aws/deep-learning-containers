import json
import os

from time import sleep, time

import pytest
import boto3
import logging
import sys

from copy import deepcopy
from invoke import run, Context

from test.test_utils import LOGGER, get_account_id_from_image_uri, get_dockerfile_path_for_image, is_dlc_cicd_context
from test.test_utils import ecr as ecr_utils
from test.test_utils.security import (
    CVESeverity,
    ScanVulnerabilityList,
    ECRScanFailureException,
    get_ecr_scan_allowlist_path,
)
from src.config import is_ecr_scan_allowlist_feature_enabled

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stderr))

MINIMUM_SEV_THRESHOLD = "HIGH"

LOWER_THRESHOLD_IMAGES = {"mxnet": ["1.8"]}


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
    repository_name = os.getenv("UPGRADE_REPO_NAME")
    ecr_account = f"{os.getenv('ACCOUNT_ID')}.dkr.ecr.{os.getenv('REGION')}.amazonaws.com"
    upgraded_image_tag = "-".join(image.replace("/", ":").split(":")[1:]) + "-up"
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
    attempt_count = 0
    apt_ran_successfully_flag = True
    while True:
        run_output = ctx.run(f"{docker_exec_cmd} {apt_command}", hide=True, warn=True)
        attempt_count += 1
        if not run_output.ok:
            LOGGER.info(
                f"Attempt no. {attempt_count} on image: {image}"
                f"Could not run apt update and upgrade. \n"
                f"Stdout is {run_output.stdout} \n"
                f"Stderr is {run_output.stderr} \n"
                f"Failed status is {run_output.exited}"
            )
            sleep(2 * 60)
        elif run_output.ok:
            break
        if attempt_count == 10:
            if not run_output.ok:
                apt_ran_successfully_flag = False
            break
    if apt_ran_successfully_flag == False:
        raise ValueError(
            f"Could not run apt update and upgrade on image: {image}. \n"
            f"Stdout is {run_output.stdout} \n"
            f"Stderr is {run_output.stderr} \n"
            f"Failed status is {run_output.exited}"
        )
    ctx.run(f"docker commit {container_id} {new_image_uri}", hide=True, warn=True)
    ctx.run(f"docker rm -f {container_id}", hide=True, warn=True)
    ctx.run(f"docker push {new_image_uri}", hide=True, warn=True)


def invoke_lambda(function_name, payload_dict={}):
    """
    Asyncronously Invokes the passed lambda.

    :param function_name: str, name of the lambda function
    :param payload_dict: dict, payload to be sent to the lambda
    """
    lambda_client = boto3.client("lambda", region_name=os.getenv("REGION"))
    response = lambda_client.invoke(
        FunctionName=function_name, InvocationType="Event", LogType="Tail", Payload=json.dumps(payload_dict)
    )
    status_code = response.get("StatusCode")
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
    file1 = open(old_filepath, "r")
    lines = file1.readlines()
    current_packages = [line.strip() for line in lines]
    package_list = current_packages + new_packages
    package_list = [f"{package_name}\n" for package_name in package_list]
    file1.close()
    run(f"rm -rf {new_filepath}")
    file2 = open(new_filepath, "w")
    file2.writelines(package_list)
    file2.close()
    s3_bucket_name = "trshanta-bucket"
    s3_client = boto3.client("s3")
    s3_client.upload_file(Filename=new_filepath, Bucket=s3_bucket_name, Key=new_filepath)


def save_scan_vulnerability_list_object_to_s3_in_json_format(image, scan_vulnerability_list_object, append_tag):
    """
    Saves the vulnerability list in the s3 bucket. It uses image to decide the name of the file on 
    the s3 bucket.

    :param image: str, image uri 
    :param vulnerability_list: ScanVulnerabilityList
    :return: str, name of the file as stored on s3
    """
    s3_bucket_name = "trshanta-bucket"
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
    """
    This method conducts the entire process that is supposed to be followed when ECR test fails. It finds all
    the fixable and non fixable vulnerabilities and all the packages that can be upgraded and finally invokes
    the Auto-Secure lambda for further processing.

    :param image: str, image uri
    :param image_allowlist: ScanVulnerabilityList, Vulnerabities that are present in the respective allowlist in the DLC git repo.
    :param ecr_image_vulnerability_list: ScanVulnerabilityList, Vulnerabities recently detected WITHOUT running apt-upgrade on the originally released image.
    :param upgraded_image_vulnerability_list: ScanVulnerabilityList, Vulnerabilites exisiting in the image WITH apt-upgrade run on it.
    :return: dict, a dictionary consisting of the entire summary of the steps run within this method.
    """
    s3_filename_for_allowlist = save_scan_vulnerability_list_object_to_s3_in_json_format(
        image, upgraded_image_vulnerability_list, "allowlist"
    )
    s3_filename_for_current_image_ecr_scan_list = save_scan_vulnerability_list_object_to_s3_in_json_format(
        image, ecr_image_vulnerability_list, "current-ecr-scanlist"
    )
    original_filepath_for_allowlist = get_ecr_scan_allowlist_path(image)
    edited_files = [{"s3_filename": s3_filename_for_allowlist, "github_filepath": original_filepath_for_allowlist}]
    vulnerabilities_fixable_by_upgrade = get_vulnerabilites_fixable_by_upgrade(
        image_allowlist, ecr_image_vulnerability_list, upgraded_image_vulnerability_list
    )
    newly_found_non_fixable_vulnerabilites = upgraded_image_vulnerability_list - image_allowlist
    fixable_list = []
    if vulnerabilities_fixable_by_upgrade:
        fixable_list = vulnerabilities_fixable_by_upgrade.vulnerability_list
    s3_filename_for_apt_upgrade_list = s3_filename_for_allowlist.replace("allowlist.json", "apt-upgrade-list.txt")
    original_filepath_for_apt_upgrade_list = "/".join(
        original_filepath_for_allowlist.split("/")[:-1] + ["apt-upgrade-list.txt"]
    )
    new_package_list = fixable_list if isinstance(fixable_list, list) else list(fixable_list.keys())
    create_and_save_package_list_to_s3(
        original_filepath_for_apt_upgrade_list, new_package_list, s3_filename_for_apt_upgrade_list
    )
    edited_files.append(
        {"s3_filename": s3_filename_for_apt_upgrade_list, "github_filepath": original_filepath_for_apt_upgrade_list}
    )
    newly_found_non_fixable_list = []
    if newly_found_non_fixable_vulnerabilites:
        newly_found_non_fixable_list = newly_found_non_fixable_vulnerabilites.vulnerability_list
    message_body = {
        "edited_files": edited_files,
        "fixable_vulnerabilities": fixable_list,
        "non_fixable_vulnerabilities": newly_found_non_fixable_list,
    }
    invoke_lambda(function_name="trshanta-ECR-AS", payload_dict=message_body)
    return_dict = deepcopy(message_body)
    return_dict["s3_filename_for_allowlist"] = s3_filename_for_allowlist
    return_dict["s3_filename_for_current_image_ecr_scan_list"] = s3_filename_for_current_image_ecr_scan_list
    return return_dict


def is_image_covered_by_allowlist_feature(image):
    """
    This method checks if the allowlist feature has been enabled for the image

    :param image: str, Image URI
    """
    if any(substring in image for substring in ["example","neuron"]):
        return False
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
    """
    For a given image it fetches all the other vulnerability lists except the vulnerability list formed by the
    ecr scan of the current image. In other words, for a given image it fetches upgraded_image_vulnerability_list and
    image_scan_allowlist.

    :param image: str Image URI for image to be tested
    :param ecr_client: boto3 Client for ECR
    :return upgraded_image_vulnerability_list: ScanVulnerabilityList, Vulnerabilites exisiting in the image WITH apt-upgrade run on it.
    :return image_allowlist: ScanVulnerabilityList, Vulnerabities that are present in the respective allowlist in the DLC git repo.
    """
    new_image_uri = get_new_image_uri(image)
    run_upgrade_on_image_and_push(image, new_image_uri)
    run_scan(ecr_client, new_image_uri)
    scan_results_with_upgrade = ecr_utils.get_ecr_image_scan_results(
        ecr_client, new_image_uri, minimum_vulnerability=MINIMUM_SEV_THRESHOLD
    )
    scan_results_with_upgrade = ecr_utils.populate_ecr_scan_with_web_scraper_results(
        new_image_uri, scan_results_with_upgrade
    )
    upgraded_image_vulnerability_list = ScanVulnerabilityList(minimum_severity=CVESeverity[MINIMUM_SEV_THRESHOLD])
    upgraded_image_vulnerability_list.construct_allowlist_from_ecr_scan_result(scan_results_with_upgrade)
    image_scan_allowlist = ScanVulnerabilityList(minimum_severity=CVESeverity[MINIMUM_SEV_THRESHOLD])
    image_scan_allowlist_path = get_ecr_scan_allowlist_path(image)
    if os.path.exists(image_scan_allowlist_path):
        image_scan_allowlist.construct_allowlist_from_file(image_scan_allowlist_path)
    return upgraded_image_vulnerability_list, image_scan_allowlist


def process_failure_routine_summary(failure_routine_summary):
    """
    This method is especially constructed to process the failure routine summary that is generated as a result of 
    calling conduct_failure_routine. It extracts lists and calls the save lists function to store them in the s3
    bucket.

    :param failure_routine_summary: dict, dictionary returned as an outcome of conduct_failure_routine method
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
    save_details.append((s3_filename_for_fixable_list, failure_routine_summary["fixable_vulnerabilities"]))
    save_details.append((s3_filename_for_non_fixable_list, failure_routine_summary["non_fixable_vulnerabilities"]))
    save_lists_in_s3(save_details)
    return s3_filename_for_fixable_list, s3_filename_for_non_fixable_list


def save_lists_in_s3(save_details, bucket_name="trshanta-bucket"):
    """
    This method takes in a list of filenames and the data corresponding to each filename and stores it in 
    the s3 bucket.

    :param save_details: list[(string, list)], a lists of tuples wherein each tuple has a filename and the corresponding data.
    :param bucket_name: string, name of the s3 bucket
    """
    s3_client = boto3.client("s3")
    for filename, data in save_details:
        with open(filename, "w") as outfile:
            json.dump(data, outfile, indent=4)
        s3_client.upload_file(Filename=filename, Bucket=bucket_name, Key=filename)


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
        s3_bucket_name = "trshanta-bucket"

        ## In case new vulnerabilities are found conduct failure routine
        newly_found_vulnerabilities = ecr_image_vulnerability_list - image_scan_allowlist
        if newly_found_vulnerabilities:
            failure_routine_summary = conduct_failure_routine(
                image, image_scan_allowlist, ecr_image_vulnerability_list, upgraded_image_vulnerability_list
            )
            s3_filename_for_fixable_list, s3_filename_for_non_fixable_list = process_failure_routine_summary(
                failure_routine_summary
            )
        assert not newly_found_vulnerabilities, (
            f"""Found {len(failure_routine_summary["fixable_vulnerabilities"])} fixable vulnerabilites """
            f"""and {len(failure_routine_summary["non_fixable_vulnerabilities"])} non fixable vulnerabilites. """
            f"""Refer to files s3://{s3_bucket_name}/{s3_filename_for_fixable_list}, s3://{s3_bucket_name}/{s3_filename_for_non_fixable_list}, """
            f"""s3://{s3_bucket_name}/{failure_routine_summary["s3_filename_for_current_image_ecr_scan_list"]} and s3://{s3_bucket_name}/{failure_routine_summary["s3_filename_for_allowlist"]}."""
        )

        ## In case there is no new vulnerability but the allowlist is outdated conduct failure routine
        vulnerabilities_that_can_be_fixed = image_scan_allowlist - upgraded_image_vulnerability_list
        if vulnerabilities_that_can_be_fixed:
            failure_routine_summary = conduct_failure_routine(
                image, image_scan_allowlist, ecr_image_vulnerability_list, upgraded_image_vulnerability_list
            )
            s3_filename_for_fixable_list, s3_filename_for_non_fixable_list = process_failure_routine_summary(
                failure_routine_summary
            )
        assert not vulnerabilities_that_can_be_fixed, (
            f"""Allowlist is Outdated!! Found {len(failure_routine_summary["fixable_vulnerabilities"])} fixable vulnerabilites """
            f"""and {len(failure_routine_summary["non_fixable_vulnerabilities"])} non fixable vulnerabilites. """
            f"""Refer to files s3://{s3_bucket_name}/{s3_filename_for_fixable_list}, s3://{s3_bucket_name}/{s3_filename_for_non_fixable_list}, """
            f"""s3://{s3_bucket_name}/{failure_routine_summary["s3_filename_for_current_image_ecr_scan_list"]} and s3://{s3_bucket_name}/{failure_routine_summary["s3_filename_for_allowlist"]}."""
        )
        return

    assert not remaining_vulnerabilities.vulnerability_list, (
        f"The following vulnerabilities need to be fixed on {image}:\n"
        f"{json.dumps(remaining_vulnerabilities.vulnerability_list, indent=4)}"
    )
