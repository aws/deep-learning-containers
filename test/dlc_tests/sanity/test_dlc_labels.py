import json

import boto3
import pytest

from test.test_utils import get_repository_and_tag_from_image_uri, LOGGER


@pytest.mark.integration("dlc_major_version_label")
def test_dlc_major_version_label(image, region):
    """
    Test to ensure that all DLC images have the LABEL "dlc_major_version"

    :param image: <str> Image URI
    :param region: <str> region where ECR repository holding the image resides
    :return:
    """
    ecr_client = boto3.client("ecr", region_name=region)

    image_repository, image_tag = get_repository_and_tag_from_image_uri(image)
    # Using "acceptedMediaTypes" on the batch_get_image request allows the returned image information to
    # provide the ECR Image Manifest in the specific format that we need, so that the image LABELS can be found
    # on the manifest. The default format does not return the image LABELs.
    response = ecr_client.batch_get_image(
        repositoryName=image_repository,
        imageIds=[{"imageTag": image_tag}],
        acceptedMediaTypes=["application/vnd.docker.distribution.manifest.v1+json"],
    )
    if not response.get("images"):
        raise KeyError(
            f"Failed to get images through ecr_client.batch_get_image response for image {image_repository}:{image_tag}"
        )
    elif not response["images"][0].get("imageManifest"):
        raise KeyError(f"imageManifest not found in ecr_client.batch_get_image response:\n{response['images']}")

    manifest_str = response["images"][0]["imageManifest"]
    # manifest_str is a json-format string
    manifest = json.loads(manifest_str)
    image_metadata = json.loads(manifest["history"][0]["v1Compatibility"])
    major_version = image_metadata["config"]["Labels"].get("dlc_major_version", None)

    assert major_version, f"{image} has no LABEL named 'dlc_major_version'. Please insert label."

    LOGGER.info(f"{image} has 'dlc_major_version' = {major_version}")
