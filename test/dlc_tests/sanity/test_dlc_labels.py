import json
import os
import re

import boto3
import pytest

from test import test_utils


@pytest.mark.integration("dlc_major_version_label")
@pytest.mark.model("N/A")
def test_dlc_major_version_label(image, region):
    """
    Test to ensure that all DLC images have the LABEL "dlc_major_version"

    :param image: <str> Image URI
    :param region: <str> region where ECR repository holding the image resides
    :return:
    """
    ecr_client = boto3.client("ecr", region_name=region)

    image_repository, image_tag = test_utils.get_repository_and_tag_from_image_uri(image)
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

    test_utils.LOGGER.info(f"{image} has 'dlc_major_version' = {major_version}")


def test_dlc_version_dockerfiles(image):
    """
    Test to make sure semantic versioning scheme in Dockerfiles is correct

    :param image: <str> ECR image URI
    """
    dlc_dir = os.getcwd().split('/test/')[0]
    job_type = test_utils.get_job_type_from_image(image)
    framework, fw_version = test_utils.get_framework_and_version_from_tag(image)
    processor = test_utils.get_processor_from_image_uri(image)

    root_dir = os.path.join(dlc_dir, framework, job_type, 'docker', fw_version)

    # Skip older FW versions that did not use this versioning scheme
    references = {
        "tensorflow2": "2.2.0",
        "tensorflow1": "1.16.0",
        "mxnet": "1.7.0",
        "pytorch": "1.5.0"
    }
    if test_utils.is_tf1(image):
        reference_fw = "tensorflow1"
    elif test_utils.is_tf2(image):
        reference_fw = "tensorflow2"
    else:
        reference_fw = framework
    if processor != "eia" and fw_version < references[reference_fw]:
        pytest.skip("Not enforcing new versioning scheme on old images")

    dockerfiles = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == f"Dockerfile.{processor}":
                dockerfile_path = os.path.join(root_dir, root, filename)
                if "example" not in dockerfile_path:
                    dockerfiles.append(dockerfile_path)

    versions = {}
    dlc_label_regex = re.compile(r'LABEL dlc_major_version="(\d+)"')
    for dockerfile in dockerfiles:
        with open(dockerfile, 'r') as df:
            for line in df:
                major_version_match = dlc_label_regex.match(line)
                if major_version_match:
                    versions[dockerfile] = int(major_version_match.group(1))

    expected_versions = []
    actual_versions = []
    possible_versions = list(range(1, len(dockerfiles) + 1))

    # Test case explicitly for TF2.3, since v2.0 is banned
    if framework == "tensorflow" and fw_version == "2.3.0":
        for version in possible_versions:
            if version >= 2:
                version += 1
            expected_versions.append(version)
    else:
        expected_versions = actual_versions

    for _, version in versions.items():
        actual_versions.append(version)

    assert sorted(actual_versions) == expected_versions, f"A major version is missing: {versions}"
