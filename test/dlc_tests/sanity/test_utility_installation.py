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


@pytest.mark.usefixtures("huggingface")
@pytest.mark.model("N/A")
@pytest.mark.integration("utility pacakges")
def test_utility_packages_using_import(training):
    """
    Verify that utility packages are installed in the Training DLC image

    :param training: training ECR image URI
    """
    ctx = Context()
    container_name = test_utils.get_container_name("utility_packages_using_import", training)
    test_utils.start_container(container_name, training, ctx)

    framework, framework_version = test_utils.get_framework_and_version_from_tag(training)
    utility_package_minimum_framework_version = {
        "mxnet": "1.8",
        "pytorch": "1.7",
        "huggingface_pytorch": "1.7",
        "tensorflow2": "2.4",
        "tensorflow1": "1.15",
        "huggingface_tensorflow": "2.4",
    }

    if framework == "tensorflow":
        framework = "tensorflow1" if framework_version.startswith("1.") else "tensorflow2"

    if Version(framework_version) < Version(utility_package_minimum_framework_version[framework]):
        pytest.skip("Extra utility packages will be added going forward.")
    
    for package in UTILITY_PACKAGES_IMPORT:
        version = test_utils.run_cmd_on_container(container_name, ctx, f"import {package}; print({package}.__version__)", executable="python").stdout.strip()
        if package == "sagemaker":
            assert Version(version) > Version("2"), f"Sagemaker version should be > 2.0. Found version {version}"


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
