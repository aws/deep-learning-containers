import os
import sys
import json
import boto3
import logging

from invoke import run
from botocore.exceptions import ClientError
from datetime import datetime, timezone

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))
LOGGER.setLevel(logging.INFO)


def pull_image_locally_with_all_its_tags_attached(image_uri):
    from test import test_utils
    run(f"docker pull {image_uri}")
    image_region = test_utils.get_region_from_image_uri(image_uri=image_uri)
    ecr_client = boto3.client("ecr", region_name=image_region)
    image_repo = image_uri.split(":")[0]
    tag_list = test_utils.get_all_the_tags_of_an_image_from_ecr(ecr_client=ecr_client, image_uri=image_uri)
    for tag in tag_list:
        run(f"docker tag {image_uri} {image_repo}:{tag}")
    return tag_list


def get_push_time_of_image_from_ecr(image_uri):
    from test import test_utils
    image_region = test_utils.get_region_from_image_uri(image_uri=image_uri)
    ecr_client = boto3.client("ecr", region_name=image_region)
    return test_utils.get_image_push_time_from_ecr(ecr_client=ecr_client, image_uri=image_uri)


def get_benchmark_tag_attached_to_the_latest_image_in_beta(autopatch_image_tag_list):
    benchmark_tag_list = [tag for tag in autopatch_image_tag_list if tag.endswith("-benchmark-tested")]
    assert len(benchmark_tag_list) == 1, f"{benchmark_tag_list} has multiple or no benchmark tested image tag"
    return benchmark_tag_list[0].replace("-autopatch","")


def get_benchmark_tested_image_uri_for_beta_image(autopatch_image_uri, benchmark_tag_in_beta):
    ap_image_repo = autopatch_image_uri.split(":")[0]
    beta_image_repo = ap_image_repo.replace("/autopatch-", "/beta-")
    return f"{beta_image_repo}:{benchmark_tag_in_beta}"


def get_image_transfer_override_flags_from_s3():
    try:
        s3_client = boto3.client("s3")
        sts_client = boto3.client("sts")
        account_id = sts_client.get_caller_identity().get("Account")
        result = s3_client.get_object(
            Bucket=f"dlc-cicd-helper-{account_id}", Key="image_transfer_override_flags.json"
        )
        json_content = json.loads(result["Body"].read().decode("utf-8"))
    except ClientError as e:
        LOGGER.error("ClientError when performing S3/STS operation. Exception: {}".format(e))
        json_content = {}
    return json_content


def is_image_transfer_enabled_by_override_flags(image_uri, image_transfer_override_flags):
    commit_id = os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION", "default")
    if commit_id in image_transfer_override_flags:
        if not image_transfer_override_flags[commit_id] or image_uri in image_transfer_override_flags[commit_id]:
            # If the list corresponding to the commit ID is empty, then return True. Otherwise, check if image uri
            # is present in the list and return True if it is present.
            LOGGER.info(f"[Override Enabled] Transfer override enabled for the image {image_uri}")
            return True
    return False


def is_image_transferable(autopatch_image_uri, beta_image_uri, image_transfer_override_flags):
    if is_image_transfer_enabled_by_override_flags(image_uri=autopatch_image_uri, image_transfer_override_flags=image_transfer_override_flags):
        return True
    beta_image_push_time = get_push_time_of_image_from_ecr(image_uri=beta_image_uri)
    current_time = datetime.now(timezone.utc)
    time_difference = current_time - beta_image_push_time
    LOGGER.info(f"Beta image was built {time_difference} ago.")
    total_time_difference_in_seconds = time_difference.total_seconds()
    if total_time_difference_in_seconds >= 3 * 24 * 60 * 60:
        # If Beta image was built more than 3 days ago
        return True

    return False

def main():
    from test import test_utils

    dlc_images = test_utils.get_dlc_images()
    image_list = dlc_images.split(" ")

    image_transfer_override_flags = get_image_transfer_override_flags_from_s3()

    for autopatch_image in image_list:
        print(autopatch_image)
        autopatch_image_tag_list = pull_image_locally_with_all_its_tags_attached(image_uri=autopatch_image)
        benchmark_tag_in_beta = get_benchmark_tag_attached_to_the_latest_image_in_beta(autopatch_image_tag_list=autopatch_image_tag_list)
        beta_latest_benchmark_image_uri = get_benchmark_tested_image_uri_for_beta_image(autopatch_image_uri=autopatch_image, benchmark_tag_in_beta=benchmark_tag_in_beta)
        beta_image_push_time = get_push_time_of_image_from_ecr(image_uri=beta_latest_benchmark_image_uri)
        retval = is_image_transferable(autopatch_image_uri=autopatch_image, beta_image_uri=beta_latest_benchmark_image_uri, image_transfer_override_flags=image_transfer_override_flags)
        print(benchmark_tag_in_beta)
        print(beta_image_push_time)
        print(type(beta_image_push_time))
        print(retval)


if __name__ == "__main__":
    main()
