from packaging.version import Version

import pytest

from invoke.context import Context

from test import test_utils


@pytest.mark.model("N/A")
@pytest.mark.integration("bokeh")
def test_bokeh(training):
    """
    Verify that bokeh is installed in the Training DLC image

    :param training: training ECR image URI
    """
    ctx = Context()
    container_name = test_utils.get_container_name("bokeh", training)
    test_utils.start_container(container_name, training, ctx)

    framework = test_utils.get_framework_from_image_uri(training)

    package_list_cmd = "conda list" if "pytorch" in framework else "pip freeze"

    test_utils.run_cmd_on_container(container_name, ctx, f"{package_list_cmd} | grep -i bokeh")
    test_utils.run_cmd_on_container(
        container_name, ctx, "import bokeh; print(bokeh.__version__)", executable="python"
    )


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


@pytest.mark.model("N/A")
@pytest.mark.integration("imageio")
def test_imageio(training):
    """
    Verify that imageio is installed in the Training DLC image

    :param training: training ECR image URI
    """
    ctx = Context()
    container_name = test_utils.get_container_name("imageio", training)
    test_utils.start_container(container_name, training, ctx)

    framework = test_utils.get_framework_from_image_uri(training)

    package_list_cmd = "conda list" if "pytorch" in framework else "pip freeze"

    test_utils.run_cmd_on_container(container_name, ctx, f"{package_list_cmd} | grep -i imageio")
    test_utils.run_cmd_on_container(
        container_name, ctx, "import imageio; print(imageio.__version__)", executable="python"
    )


@pytest.mark.model("N/A")
@pytest.mark.integration("opencv")
def test_opencv(training):
    """
    Verify that cv2 is installed in the Training DLC image

    :param training: training ECR image URI
    """
    ctx = Context()
    container_name = test_utils.get_container_name("opencv", training)
    test_utils.start_container(container_name, training, ctx)

    framework = test_utils.get_framework_from_image_uri(training)

    package_list_cmd = "conda list" if "pytorch" in framework else "pip freeze"

    test_utils.run_cmd_on_container(container_name, ctx, f"{package_list_cmd} | grep -i opencv")
    test_utils.run_cmd_on_container(container_name, ctx, "import cv2; print(cv2.__version__)", executable="python")


@pytest.mark.model("N/A")
@pytest.mark.integration("pandas")
def test_pandas(image):
    """
    It's possible that in newer python versions, we may have issues with installing pandas due to lack of presence
    of the bz2 module in py3 containers. This is a sanity test to ensure that pandas import works properly in all
    containers.

    :param image: ECR image URI
    """
    ctx = Context()
    container_name = test_utils.get_container_name("pandas", image)
    test_utils.start_container(container_name, image, ctx)

    # Make sure we can install pandas, do not fail right away if there are pip check issues
    test_utils.run_cmd_on_container(container_name, ctx, "pip install pandas", warn=True)

    pandas_import_output = test_utils.run_cmd_on_container(container_name, ctx, "import pandas", executable="python")

    assert (
        not pandas_import_output.stdout.strip()
    ), f"Expected no output when importing pandas, but got  {pandas_import_output.stdout}"

    # Simple import test to ensure we do not get a bz2 module import failure
    test_utils.run_cmd_on_container(
        container_name, ctx, "import pandas; print(pandas.__version__)", executable="python"
    )


@pytest.mark.model("N/A")
@pytest.mark.integration("plotly")
def test_plotly(training):
    """
    Verify that plotly is installed in the Training DLC image

    :param training: ECR image URI
    """
    ctx = Context()
    container_name = test_utils.get_container_name("plotly", training)
    test_utils.start_container(container_name, training, ctx)

    framework = test_utils.get_framework_from_image_uri(training)

    package_list_cmd = "conda list" if "pytorch" in framework else "pip freeze"

    test_utils.run_cmd_on_container(container_name, ctx, f"{package_list_cmd} | grep -i plotly")
    test_utils.run_cmd_on_container(
        container_name, ctx, "import plotly; print(plotly.__version__)", executable="python"
    )


@pytest.mark.model("N/A")
@pytest.mark.integration("seaborn")
def test_seaborn(training):
    """
    Verify that seaborn is installed in the Training DLC image

    :param training: ECR image URI
    """
    ctx = Context()
    container_name = test_utils.get_container_name("seaborn", training)
    test_utils.start_container(container_name, training, ctx)

    framework = test_utils.get_framework_from_image_uri(training)

    package_list_cmd = "conda list" if "pytorch" in framework else "pip freeze"

    test_utils.run_cmd_on_container(container_name, ctx, f"{package_list_cmd} | grep -i seaborn")
    test_utils.run_cmd_on_container(
        container_name, ctx, "import seaborn; print(seaborn.__version__)", executable="python"
    )


@pytest.mark.model("N/A")
@pytest.mark.integration("shap")
def test_seaborn(training):
    """
    Verify that shap is installed in the Training DLC image

    :param training: ECR image URI
    """
    ctx = Context()
    container_name = test_utils.get_container_name("shap", training)
    test_utils.start_container(container_name, training, ctx)

    framework = test_utils.get_framework_from_image_uri(training)

    package_list_cmd = "conda list" if "pytorch" in framework else "pip freeze"

    test_utils.run_cmd_on_container(container_name, ctx, f"{package_list_cmd} | grep -i shap")
    test_utils.run_cmd_on_container(
        container_name, ctx, "import shap; print(shap.__version__)", executable="python"
    )


@pytest.mark.model("N/A")
@pytest.mark.integration("sagemaker python sdk")
def test_sm_pysdk_2(training):
    """
    Simply verify that we have sagemaker > 2.0 in the python sdk.

    If you find that this test is failing because sm pysdk version is not greater than 2.0, then that means that
    the image under test needs to be updated.

    If you find that the training image under test does not have sagemaker pysdk, it should be added or explicitly
    skipped (with reasoning provided).

    :param training: training ECR image URI
    """

    _, image_framework_version = test_utils.get_framework_and_version_from_tag(training)

    if Version(image_framework_version) == Version("1.5.0"):
        pytest.skip("sagemaker version < 2.0 is installed for PT 1.5.0 images")

    # Ensure that sm py sdk 2 is on the container
    ctx = Context()
    container_name = test_utils.get_container_name("sm_pysdk", training)
    test_utils.start_container(container_name, training, ctx)

    sm_version = test_utils.run_cmd_on_container(
        container_name, ctx, "import sagemaker; print(sagemaker.__version__)", executable="python"
    ).stdout.strip()

    assert Version(sm_version) > Version("2"), f"Sagemaker version should be > 2.0. Found version {sm_version}"
