from packaging.version import Version

import pytest

from invoke.context import Context

from test import test_utils

UTILITY_PACKAGES_IMPORT = ["bokeh", "imageio", "plotly", "seaborn", "shap", "pandas", "cv2", "sagemaker"]


# TODO: Need to be added to all DLC images in furture.
@pytest.mark.usefixtures("sagemaker")
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


@pytest.mark.usefixtures("sagemaker_only", "huggingface", "non_autogluon_only")
@pytest.mark.model("N/A")
@pytest.mark.integration("utility pacakges")
def test_utility_packages_using_import(training):
    """
    Verify that utility packages are installed in the Training DLC image
    :param training: training ECR image URI
    """
    #TODO: revert once habana is supported on SM
    if "hpu" in training:
        pytest.skip("Skipping test for Habana images as SM is not yet supported")

    ctx = Context()
    container_name = test_utils.get_container_name("utility_packages_using_import", training)
    test_utils.start_container(container_name, training, ctx)

    framework, framework_version = test_utils.get_framework_and_version_from_tag(training)
    framework = framework.replace("_trcomp", "")
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

    packages_to_import = UTILITY_PACKAGES_IMPORT

    for package in packages_to_import:
        version = test_utils.run_cmd_on_container(
            container_name, ctx, f"import {package}; print({package}.__version__)", executable="python"
        ).stdout.strip()
        if package == "sagemaker":
            assert Version(version) > Version("2"), f"Sagemaker version should be > 2.0. Found version {version}"


@pytest.mark.usefixtures("sagemaker")
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

    test_utils.run_cmd_on_container(container_name, ctx, "import boto3", executable="python")


@pytest.mark.usefixtures("sagemaker")
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


@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.model("N/A")
@pytest.mark.integration("sagemaker_studio_analytics_extension")
@pytest.mark.parametrize(
    "package_name", ["pyhive", "sparkmagic", "sagemaker-studio-sparkmagic-lib", "sagemaker-studio-analytics-extension"]
)
def test_sagemaker_studio_analytics_extension(training, package_name):
    framework, framework_version = test_utils.get_framework_and_version_from_tag(training)
    utility_package_minimum_framework_version = {"pytorch": "1.7", "tensorflow": "2.4"}
    utility_package_maximum_framework_version = {"pytorch": "1.8", "tensorflow": "2.6"}
    
    if framework not in utility_package_minimum_framework_version or Version(framework_version) < Version(
        utility_package_minimum_framework_version[framework]) or Version(framework_version) > Version(
        utility_package_maximum_framework_version[framework]
    ):
        pytest.skip(f"sagemaker_studio_analytics_extension is not installed in {framework} {framework_version} DLCs")

    ctx = Context()
    container_name = test_utils.get_container_name(f"sagemaker_studio_analytics_extension-{package_name}", training)
    test_utils.start_container(container_name, training, ctx)

    # Optionally add version validation in the following steps, rather than just printing it.
    test_utils.run_cmd_on_container(container_name, ctx, f"pip list | grep -i {package_name}")
    import_package = package_name.replace("-", "_")
    import_test_cmd = (
        f"import {import_package}"
        if package_name in ["sagemaker-studio-sparkmagic-lib", "sagemaker-studio-analytics-extension"]
        else f"import {import_package}; print({import_package}.__version__)"
    )
    test_utils.run_cmd_on_container(container_name, ctx, import_test_cmd, executable="python")
