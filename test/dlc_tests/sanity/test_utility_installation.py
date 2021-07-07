from packaging.version import Version

import pytest

from invoke.context import Context

from test import test_utils

UTILITY_PACKAGES_IMPORT = [
    "bokeh",
    "imageio",
    "plotly",
    "seaborn",
    "shap",
    "sagemaker",
    "pandas",
    "cv2"
]

# TODO: Need to be added to all DLC images in furture.
@pytest.mark.model("N/A")
@pytest.mark.integration("awscli")
def test_awscli(mxnet_inference):
    """
    Ensure that boto3 is installed on mxnet inference

    :param mxnet_inference: ECR image URI
    """
    image = mxnet_inference
    ctx = Context()
    container_name = test_utils.get_container_name("awscli", image)
    test_utils.start_container(container_name, image, ctx)

    test_utils.run_cmd_on_container(container_name, ctx, "which aws")
    test_utils.run_cmd_on_container(container_name, ctx, "aws --version")


@pytest.mark.model("N/A")
@pytest.mark.integration("bokeh")
def test_utility_packages_using_import(training):
    """
    Verify that bokeh is installed in the Training DLC image

    :param training: training ECR image URI
    """
    ctx = Context()
    container_name = test_utils.get_container_name("utility_packages_using_import", training)
    test_utils.start_container(container_name, training, ctx)

    framework = test_utils.get_framework_from_image_uri(training)

    package_list_cmd = "conda list" if "pytorch" in framework else "pip freeze"
    _, image_framework_version = test_utils.get_framework_and_version_from_tag(training)
    for package in UTILITY_PACKAGES_IMPORT:
        if Version(image_framework_version) == Version("1.5.0") and package == "sagemaker":
            pytest.skip("sagemaker version < 2.0 is installed for PT 1.5.0 images")
        test_utils.run_cmd_on_container(container_name, ctx, f"{package_list_cmd} | grep -i {package}")
        version = test_utils.run_cmd_on_container(container_name, ctx, f"import {package}; print({package}.__version__)", executable="python").stdout.strip()
        if package == "sagemaker":
            assert Version(version) > Version("2"), f"Sagemaker version should be > 2.0. Found version {sm_version}"


@pytest.mark.model("N/A")
@pytest.mark.integration("boto3")
def test_boto3(mxnet_inference):
    """
    Ensure that boto3 is installed on mxnet inference

    :param mxnet_inference: ECR image URI
    """
    image = mxnet_inference
    ctx = Context()
    container_name = test_utils.get_container_name("boto3", image)
    test_utils.start_container(container_name, image, ctx)

    test_utils.run_cmd_on_container(container_name, ctx, 'import boto3', executable="python")


@pytest.mark.model("N/A")
@pytest.mark.integration("emacs")
def test_emacs(image):
    """
    Ensure that emacs is installed on every image

    :param image: ECR image URI
    """
    ctx = Context()
    container_name = test_utils.get_container_name("emacs", image)
    test_utils.start_container(container_name, image, ctx)

    # Make sure the following emacs sanity tests exit with code 0
    test_utils.run_cmd_on_container(container_name, ctx, "which emacs")
    test_utils.run_cmd_on_container(container_name, ctx, "emacs -version")
