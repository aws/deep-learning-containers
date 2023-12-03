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

from test import test_utils


def get_repository_uri(image_uri):
    """
    Returns the repository URI in the format <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/<REPOSITORY_NAME>
    """
    return image_uri.split(":")[0]


def pull_image_locally_with_all_its_tags_attached(image_uri):
    """
    Pulls the image locally and fetches all its tags and tags the image on the local host as well.

    :param image_uri: str, Image URI
    :return: List, List of all the tags attached to the image on the ECR
    """
    run(f"docker pull {image_uri}", hide=True)
    image_region = test_utils.get_region_from_image_uri(image_uri=image_uri)
    ecr_client = boto3.client("ecr", region_name=image_region)
    image_repo = image_uri.split(":")[0]
    tag_list = test_utils.get_all_the_tags_of_an_image_from_ecr(
        ecr_client=ecr_client, image_uri=image_uri
    )
    for tag in tag_list:
        run(f"docker tag {image_uri} {image_repo}:{tag}", hide=True)
    return tag_list


def is_latest_benchmark_tested_beta_image_an_autopatch_image_itself(beta_image_uri):
    """
    Checks if the latest benchmark_tested_beta_image is an autopatch image itself. It pulls the tags
    attached to the beta_benchmark_tested_image and checks for the presence of "-autopatch" in any of
    the tags.

    :param beta_image_uri: str, Image URI of beta benchmark tested image
    :return: boolean, True if beta benchmark image is an autopatch image itself. False otherwise.
    """
    image_region = test_utils.get_region_from_image_uri(image_uri=beta_image_uri)
    ecr_client = boto3.client("ecr", region_name=image_region)
    tag_list = test_utils.get_all_the_tags_of_an_image_from_ecr(
        ecr_client=ecr_client, image_uri=beta_image_uri
    )
    return any(["-autopatch" in tag for tag in tag_list])


def get_push_time_of_image_from_ecr(image_uri):
    """
    This method uses the ERC boto3 client to get the push time of an image.

    :param image_uri: str, Image URI
    :return: datetime.datetime Object, Time of the Push
    """
    image_region = test_utils.get_region_from_image_uri(image_uri=image_uri)
    ecr_client = boto3.client("ecr", region_name=image_region)
    return test_utils.get_image_push_time_from_ecr(ecr_client=ecr_client, image_uri=image_uri)


def get_benchmark_tag_attached_to_the_latest_image_in_beta(autopatch_image_tag_list):
    """
    Iterates through all the tags attached to the autopatch benchmark tested image. Filters out only the benchmark
    tested tags from the tag list and removes "-autopatch" from the tags to derive the benchmar-tested tag of Beta
    ECR.

    :param autopatch_image_tag_list: List, List of Tags attached to the AutoPatch image in the AutoPatch ECR
    :return: str, Benchmark Tag as it would look on the latest benchmark tagged image in BETA ECR.
    """
    benchmark_tag_list = [
        tag for tag in autopatch_image_tag_list if tag.endswith("-benchmark-tested")
    ]
    assert (
        len(benchmark_tag_list) == 1
    ), f"{benchmark_tag_list} has multiple or no benchmark tested image tag"
    return benchmark_tag_list[0].replace("-autopatch", "")


def get_benchmark_tested_image_uri_for_beta_image(autopatch_image_uri, benchmark_tag_in_beta):
    """
    Uses autopatch_image_uri to derive the repo for the Beta image. Appends benchmark_tag_in_beta to the beta repo
    and to form the image uri of the latest `benchmark-tested` tag image in beta repo and returns it.

    :param autopatch_image_uri: str, Image URI of AutoPatch image
    :param benchmark_tag_in_beta: str, Represents benchmark-tested tag of Beta ECRs.
    """
    autopatch_image_repo = get_repository_uri(image_uri=autopatch_image_uri)
    beta_image_repo = autopatch_image_repo.replace("/autopatch-", "/beta-")
    return f"{beta_image_repo}:{benchmark_tag_in_beta}"


def get_image_transfer_override_flags_from_s3():
    """
    Fetches the image_transfer_override_flags.json from the s3 bucket and returns its content.

    :return: dict, Contents of the image_transfer_override_flags.json
    """
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
    """
    Checks the image_transfer_override_flags to see if the current Commit ID exists as a key in the dictionary. In case
    the current commit id exists, it checks the value corresponding to the `commit-id` key and checks if it is an empty
    list or not. If it is an empty list, it means that all the images can be transferred. Thus, if the list is empty, it
    returns True - stating that the image can be transferred. However, if the list has image URIs in it, it checks if the
    image_uri passed in this method's argument is a part of thatl list or not. If it is a part of that list, returns True,
    otherwise False.

    :param image_uri: str, Image URI
    :param image_transfer_override_flags: dict, Image Override flags derived from the s3 Bucket
    """
    commit_id = os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION", "default")
    if commit_id in image_transfer_override_flags:
        if (
            not image_transfer_override_flags[commit_id]
            or image_uri in image_transfer_override_flags[commit_id]
        ):
            # If the list corresponding to the commit ID is empty, then return True. Otherwise, check if image uri
            # is present in the list and return True if it is present.
            LOGGER.info(f"[Override Enabled] Transfer override enabled for the image {image_uri}")
            return True
    return False


def transfer_image(autopatch_image_repo, autopatch_image_tag_list, beta_repo):
    """
    Transfers the image from the AutoPatch ECR to the Beta ECR. This method iterates through all the tags attached
    in the AutoPatch image and removes "-autopatch" from those tags to form the appropriate beta_tag for the beta repository.
    It then tags the image with {beta_repo}:{beta_tag} and pushes the image. It also attaches the autopatch tags to the beta image.
    In other words, it tags the image wit {beta_repo}:{autopatch_tag} and pushes the image to ensure that autopatch tags are
    preserved with the image as well. It DOES NOT transfer those autopatch_tags to the beta repo that have "benchmark-tested" in it.

    :param autopatch_image_repo: str, Image URI of the AutoPatch Image.
    :param autopatch_image_tag_list: list, List of all the image tags that were attached to the image in autopatch ECR
    :param beta_repo: str, Beta Repo
    """
    for autopatch_tag in autopatch_image_tag_list:
        beta_tag = autopatch_tag.replace("-autopatch", "")
        run(f"docker tag {autopatch_image_repo}:{autopatch_tag} {beta_repo}:{beta_tag}", hide=True)
        LOGGER.info(f"docker push {beta_repo}:{beta_tag}")
        run(f"docker push {beta_repo}:{beta_tag}", hide=True)
        if "benchmark-tested" not in autopatch_tag:
            run(
                f"docker tag {autopatch_image_repo}:{autopatch_tag} {beta_repo}:{autopatch_tag}",
                hide=True,
            )
            LOGGER.info(f"docker push {beta_repo}:{autopatch_tag}")
            run(f"docker push {beta_repo}:{autopatch_tag}", hide=True)


def is_image_transferable(autopatch_image_uri, beta_image_uri, image_transfer_override_flags):
    """
    Checks if an image is transferable or not. It first checks if the image_transfer_flag has been enabled. In case
    yes, it trasfers the image. In case not, it checks if the latest benchmark image in the Beta ECR is stale for more
    than 5 days. If yes, conducts the transfer. If not, does not conduct the transfer.

    :param autopatch_image_uri: str, Image URI of the AutoPatch image
    :param beta_image_uri: str, Image URI of the Beta image
    :param image_transfer_override_flags: dict, Image Override flags derived from the s3 Bucket
    :return: boolean, True if the transfer can happen, False otherwise.
    """
    if is_image_transfer_enabled_by_override_flags(
        image_uri=autopatch_image_uri, image_transfer_override_flags=image_transfer_override_flags
    ):
        return True
    beta_image_push_time = get_push_time_of_image_from_ecr(image_uri=beta_image_uri)
    current_time = datetime.now(timezone.utc)
    time_difference = current_time - beta_image_push_time
    LOGGER.info(f"Beta image was built {time_difference} ago.")
    total_time_difference_in_seconds = time_difference.total_seconds()
    if total_time_difference_in_seconds >= 5 * 24 * 60 * 60:
        # If Beta image was built more than 5 days ago
        return True
    elif is_latest_benchmark_tested_beta_image_an_autopatch_image_itself(
        beta_image_uri=beta_image_uri
    ):
        # If the latest benchmark tested image is autopatch image itself, transfer the image
        LOGGER.info(
            f"{autopatch_image_uri} is transferable since {beta_image_uri} is autopatch image itself."
        )
        return True

    return False


def conduct_initial_verification_to_confirm_if_image_should_be_transferred(
    autopatch_image_uri, autopatch_image_tag_list
):
    """
    This method conducts initial verification on the AutoPatch image to confirm if it belongs to AutoPatch ECR and that it
    is benchmark tested image or not.

    :param autopatch_image_uri: str, Image URI
    :param autopatch_image_tag_list: List, List of Image Tags
    """
    autopatch_repo_name, _ = test_utils.get_repository_and_tag_from_image_uri(
        image_uri=autopatch_image_uri
    )
    assert autopatch_repo_name.startswith(
        "autopatch-"
    ), f"Image {autopatch_image_uri} is not in AutoPatch ECR."

    assert any(
        [tag for tag in autopatch_image_tag_list if tag.endswith("-benchmark-tested")]
    ), f"Image {autopatch_image_uri} is not yet benchmark tested."


def main():
    """
    Driver function that handles the transfer of all the images from autopatch to beta ECRs.

    Gets the list of Image URIs that need to be transferred. Iterates through the list to check if an
    image can be transferred or not. Transfers the image in case it can be transferred.
    """

    dlc_images = test_utils.get_dlc_images()
    image_list = dlc_images.split(" ")

    ##TODO Revert:
    temp_image_repo = image_list[0].split(":")[0]
    temp_image_repo = temp_image_repo.replace("/pr-", "/autopatch-").replace(
        "/trshanta-", "/autopatch-"
    )
    temp_image_tag = "2.12.1-gpu-py310-cu118-ubuntu20.04-sagemaker-autopatch-benchmark-tested-2023-11-17-22-14-00"
    image_list = [f"{temp_image_repo}:{temp_image_tag}"]
    print(image_list)

    image_transfer_override_flags = get_image_transfer_override_flags_from_s3()

    for autopatch_image in image_list:
        LOGGER.info(f"[Processing] Image: {autopatch_image}")
        autopatch_image_tag_list = pull_image_locally_with_all_its_tags_attached(
            image_uri=autopatch_image
        )
        conduct_initial_verification_to_confirm_if_image_should_be_transferred(
            autopatch_image_uri=autopatch_image, autopatch_image_tag_list=autopatch_image_tag_list
        )
        benchmark_tag_in_beta = get_benchmark_tag_attached_to_the_latest_image_in_beta(
            autopatch_image_tag_list=autopatch_image_tag_list
        )
        beta_latest_benchmark_image_uri = get_benchmark_tested_image_uri_for_beta_image(
            autopatch_image_uri=autopatch_image, benchmark_tag_in_beta=benchmark_tag_in_beta
        )
        if is_image_transferable(
            autopatch_image_uri=autopatch_image,
            beta_image_uri=beta_latest_benchmark_image_uri,
            image_transfer_override_flags=image_transfer_override_flags,
        ):
            autopatch_image_repo = get_repository_uri(image_uri=autopatch_image)
            beta_image_repo = get_repository_uri(image_uri=beta_latest_benchmark_image_uri)
            transfer_image(
                autopatch_image_repo=autopatch_image_repo,
                autopatch_image_tag_list=autopatch_image_tag_list,
                beta_repo=beta_image_repo,
            )
            LOGGER.info(f"Transferred image {autopatch_image}")
        else:
            LOGGER.info(f"Image {autopatch_image} cannot be transferred.")


if __name__ == "__main__":
    main()
