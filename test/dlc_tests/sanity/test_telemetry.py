from test.test_utils import execute_env_variables_test, get_framework_and_version_from_tag

import pytest
from packaging.version import Version


@pytest.mark.usefixtures("sagemaker", "functionality_sanity")
@pytest.mark.model("N/A")
def test_pytorch_training_job_type_env_var(pytorch_training):
    _test_pytorch_job_type_env_var(pytorch_training, "training")


@pytest.mark.usefixtures("sagemaker", "functionality_sanity")
@pytest.mark.model("N/A")
def test_pytorch_training_job_type_env_var_arm64(pytorch_training_arm64):
    _test_pytorch_job_type_env_var(pytorch_training_arm64, "training")


@pytest.mark.usefixtures("sagemaker", "functionality_sanity")
@pytest.mark.model("N/A")
def test_pytorch_inference_job_type_env_var(pytorch_inference):
    _test_pytorch_job_type_env_var(pytorch_inference, "inference")


@pytest.mark.usefixtures("sagemaker", "functionality_sanity")
@pytest.mark.model("N/A")
def test_pytorch_inference_job_type_env_var_arm64(pytorch_inference_arm64):
    _test_pytorch_job_type_env_var(pytorch_inference_arm64, "inference")


def _test_pytorch_job_type_env_var(image, job_type):
    _, image_framework_version = get_framework_and_version_from_tag(image)
    if Version(image_framework_version) < Version("1.10"):
        pytest.skip("This env variable was added after PT 1.10 release. Skipping test.")
    if Version(image_framework_version) < Version("2.7") and (
        "graviton" in image or "arm64" in image
    ):
        pytest.skip(
            "This env variable was added for arm64 or graviton image after PT 2.7 release. Skipping test."
        )
    env_vars = {"DLC_CONTAINER_TYPE": job_type}
    container_name_prefix = (
        "pt_train_job_type_env_var" if job_type == "training" else "pt_inference_job_type_env_var"
    )
    execute_env_variables_test(
        image_uri=image,
        env_vars_to_test=env_vars,
        container_name_prefix=container_name_prefix,
    )
