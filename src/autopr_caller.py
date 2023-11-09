import json
import boto3
import os
import logging
import sys

from enum import Enum
from datetime import datetime

AUTOPR_PROD_QUEUE = "autopr-prod-queue"
S3_BUCKET = "pr-creation-data-helper"
QUEUE_REGION = "us-west-2"

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))
LOGGER.setLevel(logging.INFO)


class ImageType(str, Enum):
    INFERENCE = "inference"
    TRAINING = "training"


def get_tag_set_dictionary_from_response(tag_sets):
    tag_set_dict = {}
    for tag_set in tag_sets:
        tag_set_dict[tag_set["Key"]] = tag_set["Value"]
    return tag_set_dict


def generate_branch_name_prefix(common_image_specs: dict):
    from test import test_utils

    branch_name_prefix = f"""{common_image_specs["framework"]}-{common_image_specs["type"]}-{common_image_specs["version"]}"""
    if "customer_type" in common_image_specs:
        branch_name_prefix = f"""{branch_name_prefix}-{common_image_specs["customer_type"]}"""
    if test_utils.is_mainline_context():
        branch_name_prefix = f"{branch_name_prefix}-ML"
    elif test_utils.is_pr_context():
        branch_name_prefix = f"{branch_name_prefix}-PR"
    else:
        raise Exception("Running in an invalid environment - neither PR, not Mainline.")
    return branch_name_prefix


def get_pr_title(common_image_specs):
    title = f"""{common_image_specs["framework"].capitalize()}-{common_image_specs["type"].capitalize()}-{common_image_specs["version"]}"""
    if "customer_type" in common_image_specs:
        title = f"""{title}-{common_image_specs["customer_type"]}"""
    title = f"{title} [Patch]"
    return title


def get_image_type_from_uri(image_uri):
    return ImageType.INFERENCE if ImageType.INFERENCE in image_uri else ImageType.TRAINING


def get_same_keys_with_different_values_in_two_dictionaries(dict1: dict, dict2: dict):
    keys_with_diff_values = []
    for key_1, value_1 in dict1.items():
        if key_1 in dict2 and dict2[key_1] != value_1:
            keys_with_diff_values.append(key_1)
    return keys_with_diff_values


def remove_list_of_keys_from_dict(input_dict: dict, list_of_keys: list):
    for key in list_of_keys:
        input_dict.pop(key, None)


def get_common_image_specs_for_all_images(image_list: list):
    common_image_specs = extract_image_specs_from_image_uri(image_list[0])
    for image in image_list[1:]:
        image_specs_extracted_for_current_image = extract_image_specs_from_image_uri(
            image_uri=image
        )
        keys_with_diff_values = get_same_keys_with_different_values_in_two_dictionaries(
            common_image_specs, image_specs_extracted_for_current_image
        )
        remove_list_of_keys_from_dict(
            input_dict=common_image_specs, list_of_keys=keys_with_diff_values
        )
    return common_image_specs


def extract_image_specs_from_image_uri(image_uri):
    from test import test_utils

    images_data = {}
    (
        images_data["framework"],
        images_data["version"],
    ) = test_utils.get_framework_and_version_from_tag(image_uri)
    if test_utils.is_ec2_image(image_uri=image_uri):
        images_data["customer_type"] = "ec2"
    else:
        images_data["customer_type"] = "sagemaker"
    images_data["type"] = get_image_type_from_uri(image_uri)
    return images_data


def get_pr_body():
    from test import test_utils

    now = datetime.now()
    dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
    message = f"Update on {dt_string}"
    if test_utils.is_pr_context():
        message = f"""{message} from PR: {os.getenv("PR_NUMBER", "N/A")}"""
    elif test_utils.is_mainline_context():
        message = (
            f"""{message} from COMMIT_ID: {os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION", "N/A")}"""
        )
    return message


def get_message_body_to_be_sent_to_autopr_queue(
    branch_name_prefix,
    edited_files,
    pr_body,
    pr_title,
    repo_owner="dummyUser812",
    repo_name="deep-learning-containers",
):
    message_body = {
        "repo_owner": repo_owner,
        "repo_name": repo_name,
        "branch_name": branch_name_prefix,
        "edited_files": edited_files,
        "pr_body": pr_body,
        "pr_title": pr_title,
    }
    return message_body


def generate_edited_files_data(image_list, bucket=S3_BUCKET, folder="temp"):
    edited_files_data = []
    session = boto3.Session()

    # Then use the session to get the resource
    s3_resource = session.resource("s3")
    s3_client = s3_resource.meta.client

    my_bucket = s3_resource.Bucket(S3_BUCKET)

    for s3_object in my_bucket.objects.filter(Prefix=folder):
        response = s3_client.get_object_tagging(
            Bucket=bucket,
            Key=s3_object.key,
        )
        tag_set_dict = get_tag_set_dictionary_from_response(response["TagSet"])
        upload_path = tag_set_dict.get("upload_path", "")
        truncated_upload_path = upload_path.split("deep-learning-containers/")[-1]
        image_uri_corresponding_to_the_file = tag_set_dict.get("image_uri", "")
        if image_uri_corresponding_to_the_file not in image_list:
            continue
        if truncated_upload_path:
            edited_files_data.append(
                {
                    "s3_bucket": bucket,
                    "s3_filename": s3_object.key,
                    "github_filepath": truncated_upload_path,
                }
            )

    return edited_files_data


def send_message_to_queue(queue_name, queue_region, message_body_string):
    sqs = boto3.resource("sqs", region_name=queue_region)
    queue = sqs.get_queue_by_name(QueueName=queue_name)
    queue.send_message(MessageBody=message_body_string)


def main():
    from test import test_utils

    dlc_images = test_utils.get_dlc_images()
    image_list = dlc_images.split(" ")
    if not image_list:
        LOGGER.info(f"No image in image_list: {image_list}")
        return
    edited_files_data = generate_edited_files_data(
        image_list=image_list, folder=os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION", "temp")
    )
    ## TODO: Skip in case of no edited files
    common_image_specs = get_common_image_specs_for_all_images(image_list=image_list)
    branch_name_prefix = generate_branch_name_prefix(common_image_specs)
    pr_title = get_pr_title(common_image_specs)
    pr_body = get_pr_body()
    ## TODO: Change repo-owner
    message_body_to_be_sent_to_autopr_queue = get_message_body_to_be_sent_to_autopr_queue(
        branch_name_prefix=branch_name_prefix,
        edited_files=edited_files_data,
        pr_body=pr_body,
        pr_title=pr_title,
        repo_owner="dummyUser812",
        repo_name="deep-learning-containers",
    )
    LOGGER.info(f"Common Image Specs: {common_image_specs}")
    LOGGER.info(
        f"Message body to be sent to AutoPR Queue: {json.dumps(message_body_to_be_sent_to_autopr_queue)}"
    )
    ## TODO: only allow pr creation on mainline
    send_message_to_queue(
        queue_name=AUTOPR_PROD_QUEUE,
        queue_region=QUEUE_REGION,
        message_body_string=json.dumps(message_body_to_be_sent_to_autopr_queue),
    )


if __name__ == "__main__":
    main()
