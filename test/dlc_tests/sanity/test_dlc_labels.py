import json
import os
import re

import boto3
import pytest

from test import test_utils
from packaging.version import Version


@pytest.mark.usefixtures("sagemaker")
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


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("dlc_major_version_label")
@pytest.mark.model("N/A")
def test_dlc_major_version_dockerfiles(image):
    """
    Test to make sure semantic versioning scheme in Dockerfiles is correct

    :param image: <str> ECR image URI
    """
    dlc_dir = os.getcwd().split(f"{os.sep}test{os.sep}")[0]
    job_type = test_utils.get_job_type_from_image(image)
    framework, fw_version = test_utils.get_framework_and_version_from_tag(image)
    processor = test_utils.get_processor_from_image_uri(image)

    # Assign a string of numbers associated with python version in tag. Python major version is not sufficient to
    # define DLC major version
    python_major_minor_version = re.search(r"-py(\d{2,})", image).group(1)

    root_dir = os.path.join(dlc_dir, framework, job_type, "docker")

    # Skip older FW versions that did not use this versioning scheme
    references = {"tensorflow2": "2.2.0", "tensorflow1": "1.16.0", "mxnet": "1.7.0", "pytorch": "1.5.0"}
    if test_utils.is_tf_version("1", image):
        reference_fw = "tensorflow1"
    elif test_utils.is_tf_version("2", image):
        reference_fw = "tensorflow2"
    else:
        reference_fw = framework
    if (reference_fw in references and Version(fw_version) < Version(references[reference_fw])):
        pytest.skip(
            f"Not enforcing new versioning scheme on old image {image}. "
            f"Started enforcing version scheme on the following: {references}"
        )

    # Find all Dockerfile.<processor> for this framework/job_type's Major.Minor version
    dockerfiles = []
    fw_version_major_minor = re.match(r"(\d+\.\d+)", fw_version).group(1)
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == f"Dockerfile.{processor}":
                dockerfile_path = os.path.join(root_dir, root, filename)
                if "example" not in dockerfile_path and f"{os.sep}{fw_version_major_minor}" in dockerfile_path:
                    dockerfiles.append(dockerfile_path)

    # For the collected dockerfiles above, note the DLC major versions in each Dockerfile if python version matches
    # the current image under test
    versions = {}
    dlc_label_regex = re.compile(r'LABEL dlc_major_version="(\d+)"')
    python_version_regex = re.compile(r"ARG PYTHON_VERSION=(\d+\.\d+)")
    for dockerfile in dockerfiles:
        with open(dockerfile, "r") as df:
            dlc_version = None
            python_version = None
            for line in df:
                major_version_match = dlc_label_regex.match(line)
                python_version_match = python_version_regex.match(line)
                if major_version_match:
                    dlc_version = int(major_version_match.group(1))
                elif python_version_match:
                    python_version = python_version_match.group(1).replace(".", "")

            # Raise errors if dlc major version label and python version arg are not found in Dockerfile
            if not dlc_version:
                raise DLCMajorVersionLabelNotFound(f"Cannot find dlc_major_version label in {dockerfile}")
            if not python_version:
                raise DLCPythonVersionNotFound(f"Cannot find PYTHON_VERSION arg in {dockerfile}")
            if python_version == python_major_minor_version:
                versions[dockerfile] = dlc_version

    expected_versions = list(range(1, len(dockerfiles) + 1))
    actual_versions = sorted(versions.values())

    # Test case explicitly for TF2.3 gpu, since v1.0 is banned
    if (framework, fw_version_major_minor, processor, python_major_minor_version, job_type) == (
        "tensorflow",
        "2.3",
        "gpu",
        "37",
        "training",
    ):
        expected_versions = [v + 1 for v in expected_versions]
        assert 1 not in actual_versions, (
            f"DLC v1.0 is deprecated in TF2.3 gpu containers, but found major version 1 "
            f"in one of the Dockerfiles. Please inspect {versions}"
        )

    # Test case explicitly for PyTorch 1.6.0 training gpu, since v2.0 is banned
    if (framework, fw_version_major_minor, processor, python_major_minor_version, job_type) == (
        "pytorch",
        "1.6",
        "gpu",
        "36",
        "training",
    ):
        expected_versions = [v + 1 for v in expected_versions]
        expected_versions[0] = 1
        assert 2 not in actual_versions, (
            f"DLC v2.0 is deprecated in PyTorch 1.6.0 gpu containers, but found major version 2 "
            f"in one of the Dockerfiles. Please inspect {versions}"
        )

    # Note: If, for example, we find 3 dockerfiles with the same framework major/minor version, same processor,
    # and same python major/minor version, we will expect DLC major versions 1, 2, and 3. If an exception needs to be
    # made to this rule, please see the above handling of TF2.3 as an example.
    assert actual_versions == expected_versions, (
        f"Found DLC major versions {actual_versions} but expected {expected_versions} for "
        f"{framework} {job_type} {processor}. Full version info: {versions}. Py version: {python_major_minor_version}"
    )


class DLCMajorVersionLabelNotFound(Exception):
    pass


class DLCPythonVersionNotFound(Exception):
    pass
