import os
import re

import pytest

from test import test_utils
from packaging.specifiers import SpecifierSet
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
    labels = test_utils.get_labels_from_ecr_image(image, region)
    major_version = labels.get("dlc_major_version", None)

    assert major_version, f"{image} has no LABEL named 'dlc_major_version'. Please insert label."

    test_utils.LOGGER.info(f"{image} has 'dlc_major_version' = {major_version}")


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("dlc_labels")
@pytest.mark.model("N/A")
def test_dlc_standard_labels(image, region):
    customer_type_label_prefix = "ec2" if test_utils.is_ec2_image(image) else "sagemaker"

    framework, fw_version = test_utils.get_framework_and_version_from_tag(image)
    framework = framework.replace("_", "-")
    fw_version = fw_version.replace(".", "-")
    device_type = test_utils.get_processor_from_image_uri(image)
    if device_type == "gpu":
        cuda_verison = test_utils.get_cuda_version_from_tag(image)
        device_type = f"{device_type}.{cuda_verison}"
    python_version = test_utils.get_python_version_from_image_uri(image)
    job_type = test_utils.get_job_type_from_image(image)
    transformers_version = test_utils.get_transformers_version_from_image_uri(image).replace(
        ".", "-"
    )
    os_version = test_utils.get_os_version_from_image_uri(image).replace(".", "-")

    # TODO: Add x86 env variable to check explicitly for x86, instead of assuming that everything not graviton is x86
    arch_type = "graviton" if test_utils.is_graviton_architecture() else "x86"

    contributor = test_utils.get_contributor_from_image_uri(image)

    expected_labels = [
        f"com.amazonaws.ml.engines.{customer_type_label_prefix}.dlc.framework.{framework}.{fw_version}",
        f"com.amazonaws.ml.engines.{customer_type_label_prefix}.dlc.device.{device_type}",
        f"com.amazonaws.ml.engines.{customer_type_label_prefix}.dlc.python.{python_version}",
        f"com.amazonaws.ml.engines.{customer_type_label_prefix}.dlc.job.{job_type}",
        f"com.amazonaws.ml.engines.{customer_type_label_prefix}.dlc.arch.{arch_type}",
        f"com.amazonaws.ml.engines.{customer_type_label_prefix}.dlc.os.{os_version}",
    ]

    if contributor:
        expected_labels.append(
            f"com.amazonaws.ml.engines.{customer_type_label_prefix}.dlc.contributor.{contributor}"
        )
    if transformers_version:
        expected_labels.append(
            f"com.amazonaws.ml.engines.{customer_type_label_prefix}.dlc.lib.transformers.{transformers_version}"
        )

    actual_labels = test_utils.get_labels_from_ecr_image(image, region)

    missing_labels = []

    for label in expected_labels:
        if label not in actual_labels:
            missing_labels.append(label)

    # TODO: Remove this when ec2 labels are added. For now, ensure they are not added.
    if customer_type_label_prefix == "ec2":
        assert set(missing_labels) == set(expected_labels), (
            f"EC2 labels are not supported yet, and should not be added to containers. "
            f"{set(expected_labels) - set(missing_labels)} should not be present."
        )
    else:
        assert not missing_labels, (
            f"Labels {missing_labels} are expected in image {image}, but cannot be found. "
            f"All labels on image: {actual_labels}"
        )


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("dlc_labels")
@pytest.mark.model("N/A")
def test_max_sagemaker_labels(image, region):
    # Max ml engines sagemaker labels allowed:
    max_labels = 10

    actual_labels = test_utils.get_labels_from_ecr_image(image, region)

    standard_labels = []
    for label in actual_labels:
        if label.startswith("com.amazonaws.ml.engines.sagemaker"):
            standard_labels.append(label)

    standard_label_count = len(standard_labels)
    assert standard_label_count <= max_labels, (
        f"Max of {max_labels} labels are supported. "
        f"Currently there are {standard_label_count}: {standard_labels}"
    )


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
    if "neuron" in image:
        neuron_sdk_version = test_utils.get_neuron_sdk_version_from_tag(image)

    # TODO: Expected dockerfiles does not properly handle multiple python versions. We will fix this separately, and skip for the
    # eia condition in the interim to unblock the release.
    if processor == "eia":
        pytest.skip(
            "Temporarily skip EIA because of lack of multiple python version support for the same framework version"
        )

    # Assign a string of numbers associated with python version in tag. Python major version is not sufficient to
    # define DLC major version
    python_major_minor_version = re.search(r"-py(\d{2,})", image).group(1)

    root_dir = os.path.join(dlc_dir, framework, job_type, "docker")

    # Skip older FW versions that did not use this versioning scheme
    references = {
        "tensorflow2": "2.2.0",
        "tensorflow1": "1.16.0",
        "mxnet": "1.7.0",
        "pytorch": "1.5.0",
    }
    excluded_versions = {
        "tensorflow2": [SpecifierSet("<2.2")],
        "tensorflow1": [SpecifierSet("<1.16")],
        "mxnet": [SpecifierSet("<1.7")],
        # HACK Temporary exception PT 1.11 and PT 1.12 since they use different cuda versions for ec2 and SM
        "pytorch": [SpecifierSet("<1.5"), SpecifierSet("==1.11.*"), SpecifierSet("==1.12.*")],
        # autogluon 0.7.0 has v1 and v2; v1 has deprecation-path MMS serving fallback option; v2 is the main version based on torch-serve
        "autogluon": [SpecifierSet("==0.7.*")],
    }
    if test_utils.is_tf_version("1", image):
        reference_fw = "tensorflow1"
    elif test_utils.is_tf_version("2", image):
        reference_fw = "tensorflow2"
    else:
        reference_fw = framework
    if any(Version(fw_version) in ev for ev in excluded_versions.get(reference_fw, [])):
        pytest.skip(
            f"Not enforcing new versioning scheme on old image {image}. "
            f"Enforcing version scheme on {reference_fw} versions that do not match {excluded_versions[reference_fw]}"
        )

    # Find all Dockerfile.<processor> for this framework/job_type's Major.Minor version
    dockerfiles = []
    fw_version_major_minor = re.match(r"(\d+\.\d+)", fw_version).group(1)
    dockerfiles_of_interest = test_utils.get_expected_dockerfile_filename(processor, image)
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == dockerfiles_of_interest:
                dockerfile_path = os.path.join(root_dir, root, filename)
                if f"{os.sep}{fw_version_major_minor}" in dockerfile_path:
                    if "neuron" in image:
                        if "sdk" + neuron_sdk_version == os.path.basename(root):
                            dockerfiles.append(dockerfile_path)
                    elif "example" not in dockerfile_path:
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
                raise DLCMajorVersionLabelNotFound(
                    f"Cannot find dlc_major_version label in {dockerfile}"
                )
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
        f"{framework} {job_type} {processor}. Full version info: {versions}. Py version: {python_major_minor_version}. "
        f"Dockerfiles looked into: {dockerfiles}"
    )


@pytest.mark.skipif(
    not test_utils.is_mainline_context(),
    reason="This test only applies to Release Candidate images",
)
@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("dlc_nightly_feature_label")
@pytest.mark.model("N/A")
def test_dlc_nightly_feature_labels(image, region):
    """
    Test to ensure that nightly feature labels are not applied on prod DLCs
    :param image:
    :return:
    """
    image_labels = test_utils.get_labels_from_ecr_image(image, region)
    nightly_labels_on_image = [
        label.value
        for label in test_utils.NightlyFeatureLabel.__members__.values()
        if label.value in image_labels
    ]
    assert not nightly_labels_on_image, (
        f"The labels {nightly_labels_on_image} must not be applied on images built to be release candidates.\n"
        f"These labels are currently applied on {image}, and must be removed to proceed further."
    )


class DLCMajorVersionLabelNotFound(Exception):
    pass


class DLCPythonVersionNotFound(Exception):
    pass
