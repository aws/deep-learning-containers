import pytest

from packaging.version import Version

from test.test_utils import (
    get_framework_and_version_from_tag,
    execute_env_variables_test
)


@pytest.mark.usefixtures("sagemaker", "functionality_sanity")
@pytest.mark.model("N/A")
def test_pytorch_training_job_type_env_var(pytorch_training):
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_training)
    if Version(image_framework_version) < Version("1.10"):
        pytest.skip("This env variable was added after PT 1.10 release. Skipping test.")
    env_vars = {"DLC_CONTAINER_TYPE": "training"}
    container_name_prefix = "pt_train_job_type_env_var"
    execute_env_variables_test(
        image_uri=pytorch_training,
        env_vars_to_test=env_vars,
        container_name_prefix=container_name_prefix,
    )


@pytest.mark.usefixtures("sagemaker", "functionality_sanity")
@pytest.mark.model("N/A")
def test_pytorch_inference_job_type_env_var(pytorch_inference):
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_inference)
    if Version(image_framework_version) < Version("1.10"):
        pytest.skip("This env variable was added after PT 1.10 release. Skipping test.")
    env_vars = {"DLC_CONTAINER_TYPE": "inference"}
    container_name_prefix = "pt_inference_job_type_env_var"
    execute_env_variables_test(
        image_uri=pytorch_inference,
        env_vars_to_test=env_vars,
        container_name_prefix=container_name_prefix,
    )
