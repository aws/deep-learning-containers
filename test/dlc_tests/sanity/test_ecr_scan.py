import json
import os
from time import sleep
import boto3

import pytest
import traceback
import copy

from invoke import run, Context
from packaging.version import Version
from packaging.specifiers import SpecifierSet
from test import test_utils

from test.test_utils import (
    LOGGER,
    EnhancedJSONEncoder,
    get_account_id_from_image_uri,
    get_framework_and_version_from_tag,
    get_repository_and_tag_from_image_uri,
    get_all_the_tags_of_an_image_from_ecr,
    is_huggingface_image,
    is_image_available_locally,
    login_to_ecr_registry,
    get_region_from_image_uri,
    ECR_ENHANCED_SCANNING_REPO_NAME,
    ECR_ENHANCED_REPO_REGION,
    is_generic_image,
    get_allowlist_path_for_enhanced_scan_from_env_variable,
    get_ecr_scan_allowlist_path,
    get_sha_of_an_image_from_ecr,
    is_mainline_context,
    is_test_phase,
)
from test.test_utils import ecr as ecr_utils
from test.test_utils.security import (
    CVESeverity,
    ECREnhancedScanVulnerabilityList,
    get_target_image_uri_using_current_uri_and_target_repo,
    wait_for_enhanced_scans_to_complete,
    extract_non_patchable_vulnerabilities,
    generate_future_allowlist,
)
from src.config import is_ecr_scan_allowlist_feature_enabled
from src import utils as src_utils
from src.codebuild_environment import get_cloned_folder_path

ALLOWLIST_FEATURE_ENABLED_IMAGES = {"mxnet": SpecifierSet(">=1.8.0,<1.9.0")}


def is_image_covered_by_allowlist_feature(image):
    """
    This method checks if the allowlist feature has been enabled for the image

    :param image: str, Image URI
    """
    image_framework, image_version = get_framework_and_version_from_tag(image)
    if image_framework not in ALLOWLIST_FEATURE_ENABLED_IMAGES or any(
        substring in image for substring in ["example"]
    ):
        return False
    if Version(image_version) in ALLOWLIST_FEATURE_ENABLED_IMAGES[image_framework]:
        return True
    return False


def get_minimum_sev_threshold_level(image):
    """
    This method gets the value for minimum threshold level. This threshold level determines the
    vulnerability severity above which we want to raise an alarm.

    :param image: str Image URI for which threshold has to be set
    """
    if is_huggingface_image():
        return "CRITICAL"
    if is_generic_image():
        return "HIGH"
    if is_image_covered_by_allowlist_feature(image):
        return "MEDIUM"
    return "HIGH"


def upload_json_to_image_data_storage_s3_bucket(
    ecr_client_for_enhanced_scanning_repo, ecr_enhanced_repo_uri, upload_list
):
    """
    This method retrieves the unique identifier of the image from ECR and uses it as the directory to upload the data in
    provided `upload_list` to the image-data-storage s3 bucket.
    This information is used for the scanning dashboard to identify image allowlist information.

    :param ecr_client_for_enhanced_scanning_repo: boto3 ecr client
    :param ecr_enhanced_repo_uri: str, image's uri on ecr enhanced repo
    :param upload_list: list of dictionaries with s3_filename, name of file to upload, and upload_data, data to upload as keys
    """
    image_sha = get_sha_of_an_image_from_ecr(
        ecr_client_for_enhanced_scanning_repo, ecr_enhanced_repo_uri
    )
    s3_resource = boto3.resource("s3")
    sts_client = boto3.client("sts")
    account_id = os.getenv("AUTOPATCH_STORAGE_ACCOUNT") or sts_client.get_caller_identity().get(
        "Account"
    )
    for to_upload in upload_list:
        s3_filepath = f"{image_sha}/{to_upload['s3_filename']}"
        upload_data = json.dumps(
            to_upload["upload_data"],
            indent=4,
            cls=EnhancedJSONEncoder,
        )
        s3object = s3_resource.Object(f"image-data-storage-{account_id}", s3_filepath)
        s3object.put(Body=(bytes(upload_data.encode("UTF-8"))))
        LOGGER.info(
            f"{to_upload['s3_filename']} uploaded to image-data-storage s3 bucket at {s3_filepath}"
        )


def add_core_packages_to_upload_list_if_exists(image, upload_list):
    """
    This method retrieves the image's corresponding core package data from the repo if it exists and adds it to the upload_list
    with s3_filename as core_packages.json.

    :param image: str, the corresponding image
    :param upload_list: list of dictionaries with s3_filename, name of file to upload, and upload_data, data to upload as keys
    """
    core_packages_path = src_utils.get_core_packages_path(image)
    if not os.path.exists(core_packages_path):
        return
    with open(core_packages_path, "r") as f:
        upload_list.append({"s3_filename": "core_packages.json", "upload_data": json.load(f)})


def conduct_preprocessing_of_images_before_running_ecr_scans(image, ecr_client, sts_client, region):
    """
    Conducts the following steps before starting any kind of ecr test:
        1. Pulls the image in case it is not existing locally.
        2. Pulls all the additional tags of the image.
        3. In case the test_account_id != image_account_id it reuploads the pulled images to the test_account_id for conducting the tests.
           Therafter, it replaces the original image uri with the new one, to the one that points to the image in test_account_id, and returns
           new image id.

    :param image: str, Image URI for image to be tested
    :param ecr_client: boto3 Client for ECR
    :param sts_client: boto3 Client for STS
    :param region: str, Name of region where test is executed
    :return image: str, Image URI for image to be tested. If test_account_id is same as image_accout_id, the image uri remains same as input param
                        otherwise, a new image uri is returned.
    """
    test_account_id = sts_client.get_caller_identity().get("Account")
    image_account_id = get_account_id_from_image_uri(image)
    image_region = get_region_from_image_uri(image)
    image_repo_name, original_image_tag = get_repository_and_tag_from_image_uri(image)
    additional_image_tags = get_all_the_tags_of_an_image_from_ecr(ecr_client, image)

    if not is_image_available_locally(image):
        LOGGER.info(f"Image {image} not available locally!! Pulling the image...")
        login_to_ecr_registry(Context(), image_account_id, image_region)
        run(f"docker pull {image}")
        if not is_image_available_locally(image):
            raise RuntimeError("Image shown as not available even after pulling")

    for additional_tag in additional_image_tags:
        image_uri_with_new_tag = image.replace(original_image_tag, additional_tag)
        run(f"docker tag {image} {image_uri_with_new_tag}", hide=True)

    if image_account_id != test_account_id:
        original_image = image
        target_image_repo_name = f"beta-{image_repo_name}"
        for additional_tag in additional_image_tags:
            image_uri_with_new_tag = original_image.replace(original_image_tag, additional_tag)
            new_image_uri = ecr_utils.reupload_image_to_test_ecr(
                image_uri_with_new_tag, target_image_repo_name, region
            )
            if image_uri_with_new_tag == original_image:
                image = new_image_uri

    return image


def helper_function_for_leftover_vulnerabilities_from_enhanced_scanning(
    image,
    python_version=None,
    remove_non_patchable_vulns=False,
    minimum_sev_threshold=None,
    allowlist_removal_enabled=True,
):
    """
    Acts as a helper function that conducts enhanced scan on an image URI and then returns the list of leftover vulns
    after removing the allowlisted vulns.
    1. Upload image to the ECR Enhanced Scanning Testing Repo.
    2. Wait for the scans to complete - takes approx 10 minutes for big images. Once the scan is complete,
        the scan status changes to ACTIVE
    3. If the status does not turn to ACTIVE, raise a TimeOut Error
    4. Read the ecr_scan_results and remove the allowlisted vulnerabilities from it
    5. Return the leftover list

    :param image: str Image URI for image to be tested
    :param python_version: str, This parameter is used for extracting allowlist for canary image uris that do not have a python version in it.
    :param remove_non_patchable_vulns: boolean, This parameter tells the method if it should remove non-patchable vulns or not. In case set to True, the non-patchable vulns will be removed.
    :param minimum_sev_threshold: str, If minimum_sev_threshold is set vulnerabilities with severity < minimum_sev_threshold will not be taken into consideration.
    :param allowlist_removal_enabled: boolean, Value of this parameter decides if we should remove allowlisted vulnearbilities from the scanner results.
    :return: remaining_vulnerabilities, ECREnhancedScanVulnerabilityList Object with leftover vulnerability data
    :return: ecr_enhanced_repo_uri, String for the image uri in the enhanced scanning repo
    """
    ecr_enhanced_repo_uri = get_target_image_uri_using_current_uri_and_target_repo(
        image,
        target_repository_name=ECR_ENHANCED_SCANNING_REPO_NAME,
        target_repository_region=ECR_ENHANCED_REPO_REGION,
        append_tag="ENHSCAN",
    )

    # ecr_enhanced_repo_uri for Huggingface Neuron images tends to be greater than 128 in length and leads to docker tag failures.
    # The if condition below edits the tag to use short names instead of long ones.
    if all(temp_string in image for temp_string in ["huggingface", "neuron"]):
        ecr_enhanced_repo_uri = ecr_enhanced_repo_uri.replace("-huggingface-", "-hf-")
        ecr_enhanced_repo_uri = ecr_enhanced_repo_uri.replace("-pytorch-", "-pt-")
        ecr_enhanced_repo_uri = ecr_enhanced_repo_uri.replace("-tensorflow-", "-tf-")

    run(f"docker tag {image} {ecr_enhanced_repo_uri}", hide=True)
    ecr_utils.reupload_image_to_test_ecr(
        ecr_enhanced_repo_uri,
        ECR_ENHANCED_SCANNING_REPO_NAME,
        ECR_ENHANCED_REPO_REGION,
        pull_image=False,
    )

    ecr_client_for_enhanced_scanning_repo = boto3.client(
        "ecr", region_name=ECR_ENHANCED_REPO_REGION
    )
    wait_for_enhanced_scans_to_complete(
        ecr_client_for_enhanced_scanning_repo, ecr_enhanced_repo_uri
    )
    LOGGER.info(f"finished wait_for_enhanced_scans_to_complete, {image}")
    sleep(1 * 60)

    scan_results = ecr_utils.get_all_ecr_enhanced_scan_findings(
        ecr_client=ecr_client_for_enhanced_scanning_repo, image_uri=ecr_enhanced_repo_uri
    )
    scan_results = json.loads(json.dumps(scan_results, cls=EnhancedJSONEncoder))

    minimum_sev_threshold = minimum_sev_threshold or get_minimum_sev_threshold_level(image)
    ecr_image_vulnerability_list = ECREnhancedScanVulnerabilityList(
        minimum_severity=CVESeverity[minimum_sev_threshold]
    )
    ecr_image_vulnerability_list.construct_allowlist_from_ecr_scan_result(scan_results)

    image_scan_allowlist = ECREnhancedScanVulnerabilityList(
        minimum_severity=CVESeverity[minimum_sev_threshold]
    )
    common_allowlist = ECREnhancedScanVulnerabilityList(
        minimum_severity=CVESeverity[minimum_sev_threshold]
    )
    common_allowlist_path = os.path.join(
        os.sep, get_cloned_folder_path(), "data", "common-ecr-scan-allowlist.json"
    )

    try:
        # Derive Image Scan Allowlist Path
        if is_generic_image():
            image_scan_allowlist_path = get_allowlist_path_for_enhanced_scan_from_env_variable()
        else:
            image_scan_allowlist_path = get_ecr_scan_allowlist_path(
                image, python_version=python_version
            )
        LOGGER.info(f"[Allowlist] Trying to locate Allowlist at PATH: {image_scan_allowlist_path}")
        # Check if image Scan Allowlist Path exists
        if os.path.exists(image_scan_allowlist_path) and allowlist_removal_enabled:
            image_scan_allowlist.construct_allowlist_from_file(image_scan_allowlist_path)
            LOGGER.info(
                f"[Allowlist] Using allowlist at location {image_scan_allowlist_path} to skip {image_scan_allowlist.get_summarized_info()}"
            )
    except:
        LOGGER.info(f"[Allowlist] Image scan allowlist path could not be derived for {image}")
        traceback.print_exc()

    if (
        allowlist_removal_enabled
        and os.path.exists(common_allowlist_path)
        and not is_generic_image()
    ):
        common_allowlist.construct_allowlist_from_file(common_allowlist_path)
        image_scan_allowlist = image_scan_allowlist + common_allowlist
        LOGGER.info(
            f"[Common Allowlist] Extracted common allowlist from {common_allowlist_path} with vulns: {common_allowlist.get_summarized_info()}"
        )

    remaining_vulnerabilities = ecr_image_vulnerability_list - image_scan_allowlist
    LOGGER.info(f"ECR Enhanced Scanning test completed for image: {image}")
    allowlist_for_daily_scans = image_scan_allowlist

    if remove_non_patchable_vulns:
        non_patchable_vulnerabilities = ECREnhancedScanVulnerabilityList(
            minimum_severity=CVESeverity[minimum_sev_threshold]
        )

        ## non_patchable_vulnerabilities is a subset of remaining_vulnerabilities that cannot be patched.
        ## Thus, if remaining_vulnerabilities exists, we need to find the non_patchable_vulnerabilities from it.
        if remaining_vulnerabilities:
            non_patchable_vulnerabilities = extract_non_patchable_vulnerabilities(
                remaining_vulnerabilities, ecr_enhanced_repo_uri
            )

        future_allowlist = generate_future_allowlist(
            ecr_image_vulnerability_list=ecr_image_vulnerability_list,
            image_scan_allowlist=image_scan_allowlist,
            non_patchable_vulnerabilities=non_patchable_vulnerabilities,
        )
        allowlist_for_daily_scans = future_allowlist

        # Note that ecr_enhanced_repo_uri will point to enhanced scan repo, thus we use image in the unique_s3 function below
        # as we want to upload the allowlist to the location that has repo of the actual image.
        future_allowlist_upload_path = (
            src_utils.get_unique_s3_path_for_uploading_data_to_pr_creation_bucket(
                image_uri=image, file_name="future_os_scan_allowlist.json"
            )
        )
        upload_tag_set = [
            {
                "Key": "upload_path",
                "Value": src_utils.remove_repo_root_folder_path_from_the_given_path(
                    given_path=image_scan_allowlist_path
                ),
            },
            {"Key": "image_uri", "Value": image},
        ]
        src_utils.upload_data_to_pr_creation_s3_bucket(
            upload_data=json.dumps(
                future_allowlist.vulnerability_list, indent=4, cls=test_utils.EnhancedJSONEncoder
            ),
            s3_filepath=future_allowlist_upload_path,
            tag_set=upload_tag_set,
        )

        if remaining_vulnerabilities:
            remaining_vulnerabilities = remaining_vulnerabilities - non_patchable_vulnerabilities

        LOGGER.info(
            f"[FutureAllowlist][image_uri:{ecr_enhanced_repo_uri}] {json.dumps(future_allowlist.vulnerability_list, cls= test_utils.EnhancedJSONEncoder)}"
        )
        LOGGER.info(
            f"[NonPatchableVulns] [image_uri:{ecr_enhanced_repo_uri}] {json.dumps(non_patchable_vulnerabilities.vulnerability_list, cls= test_utils.EnhancedJSONEncoder)}"
        )

    if is_mainline_context() and is_test_phase() and not is_generic_image():
        upload_list = [
            {
                "s3_filename": "ecr_allowlist.json",
                "upload_data": allowlist_for_daily_scans.vulnerability_list,
            }
        ]
        add_core_packages_to_upload_list_if_exists(image, upload_list)
        upload_json_to_image_data_storage_s3_bucket(
            ecr_client_for_enhanced_scanning_repo, ecr_enhanced_repo_uri, upload_list
        )

    return remaining_vulnerabilities, ecr_enhanced_repo_uri


@pytest.mark.usefixtures("sagemaker", "security_sanity")
@pytest.mark.model("N/A")
@pytest.mark.integration("ECR Enhanced Scans on Images")
def test_ecr_enhanced_scan(image, ecr_client, sts_client, region):
    """
    Run ECR Enhanced Scan Tool on an image being tested, and raise Error if vulnerabilities found
    1. Use helper_function_for_leftover_vulnerabilities_from_enhanced_scanning to get the list of vulnerabilities
    2. In case any vulnerability is remaining after removal, raise an error

    :param image: str Image URI for image to be tested
    :param ecr_client: boto3 Client for ECR
    :param sts_client: boto3 Client for STS
    :param region: str Name of region where test is executed
    """
    LOGGER.info(f"Running test_ecr_enhanced_scan for image {image}")
    image = conduct_preprocessing_of_images_before_running_ecr_scans(
        image, ecr_client, sts_client, region
    )

    (
        remaining_vulnerabilities,
        _,
    ) = helper_function_for_leftover_vulnerabilities_from_enhanced_scanning(
        image, remove_non_patchable_vulns="autopatch" in image
    )
    if remaining_vulnerabilities:
        LOGGER.info(
            f"Total of {len(remaining_vulnerabilities.vulnerability_list)} vulnerabilities need to be fixed on {image}:\n"
            f"{json.dumps(remaining_vulnerabilities.vulnerability_list, cls= test_utils.EnhancedJSONEncoder)}"
        )

        assert not remaining_vulnerabilities.vulnerability_list, (
            f"Total of {len(remaining_vulnerabilities.vulnerability_list)} vulnerabilities need to be fixed on {image}:\n"
            f"{json.dumps(remaining_vulnerabilities.vulnerability_list, cls= test_utils.EnhancedJSONEncoder)}"
        )
