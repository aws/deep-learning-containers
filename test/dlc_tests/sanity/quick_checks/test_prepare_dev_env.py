import pytest
import os

from unittest.mock import patch, mock_open
from src import prepare_dlc_dev_environment
from test.test_utils import is_pr_context


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("build_frameworks")
@pytest.mark.skipif(
    not is_pr_context(),
    reason="Development environment utility only needs to be tested in PRs, and does not add functional value in other contexts.",
)
def test_build_frameworks():
    overrider = prepare_dlc_dev_environment.TomlOverrider()
    overrider.set_build_frameworks(("pytorch", "tensorflow"))

    assert overrider.overrides["build"]["build_frameworks"] == ["pytorch", "tensorflow"]


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("job_types")
@pytest.mark.skipif(
    not is_pr_context(),
    reason="Development environment utility only needs to be tested in PRs, and does not add functional value in other contexts.",
)
def test_build_job_types():
    overrider = prepare_dlc_dev_environment.TomlOverrider()
    overrider.set_job_type(("inference", "training"))
    assert (
        overrider.overrides["build"]["build_training"] is True
        and overrider.overrides["build"]["build_inference"] is True
    )

    overrider.set_job_type(["inference"])
    assert (
        overrider.overrides["build"]["build_training"] is False
        and overrider.overrides["build"]["build_inference"] is True
    )

    overrider.set_job_type(["training"])
    assert (
        overrider.overrides["build"]["build_training"] is True
        and overrider.overrides["build"]["build_inference"] is False
    )

    overrider.set_job_type([])
    assert (
        overrider.overrides["build"]["build_training"] is False
        and overrider.overrides["build"]["build_inference"] is False
    )


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("test_types")
@pytest.mark.skipif(
    not is_pr_context(),
    reason="Development environment utility only needs to be tested in PRs, and does not add functional value in other contexts.",
)
def test_set_test_types():
    overrider = prepare_dlc_dev_environment.TomlOverrider()

    # Test case with a subset of test types
    test_types = ["ec2_tests", "ecs_tests", "sagemaker_remote_tests"]
    overrider.set_test_types(test_types)
    assert overrider.overrides["test"]["sanity_tests"] is False
    assert overrider.overrides["test"]["security_tests"] is False
    assert overrider.overrides["test"]["ecs_tests"] is True
    assert overrider.overrides["test"]["eks_tests"] is False
    assert overrider.overrides["test"]["ec2_tests"] is True
    assert overrider.overrides["test"]["sagemaker_local_tests"] is False
    assert overrider.overrides["test"]["sagemaker_remote_tests"] is True

    # Test case with no test types (default behavior); Should not override anything
    empty_overrider = prepare_dlc_dev_environment.TomlOverrider()
    empty_test_types = []
    empty_overrider.set_test_types(empty_test_types)
    assert not empty_overrider.overrides["test"]


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("dev_mode")
@pytest.mark.skipif(
    not is_pr_context(),
    reason="Development environment utility only needs to be tested in PRs, and does not add functional value in other contexts.",
)
def test_set_dev_mode():
    overrider = prepare_dlc_dev_environment.TomlOverrider()

    # test with no dev mode provided
    overrider.set_dev_mode(None)
    assert overrider.overrides["dev"]["graviton_mode"] is False
    assert overrider.overrides["dev"]["neuronx_mode"] is False
    assert overrider.overrides["dev"]["deep_canary_mode"] is False

    overrider.set_dev_mode("graviton_mode")
    assert overrider.overrides["dev"]["graviton_mode"] is True
    assert overrider.overrides["dev"]["neuronx_mode"] is False
    assert overrider.overrides["dev"]["deep_canary_mode"] is False

    overrider.set_dev_mode("neuronx_mode")
    assert overrider.overrides["dev"]["graviton_mode"] is False
    assert overrider.overrides["dev"]["neuronx_mode"] is True
    assert overrider.overrides["dev"]["deep_canary_mode"] is False

    overrider.set_dev_mode("deep_canary_mode")
    assert overrider.overrides["dev"]["graviton_mode"] is False
    assert overrider.overrides["dev"]["neuronx_mode"] is False
    assert overrider.overrides["dev"]["deep_canary_mode"] is True

    # Test case with multiple dev modes (error)
    with pytest.raises(ValueError):
        overrider.set_dev_mode(["graviton_mode", "neuronx_mode"])


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("set_buildspec")
@pytest.mark.skipif(
    not is_pr_context(),
    reason="Development environment utility only needs to be tested in PRs, and does not add functional value in other contexts.",
)
def test_set_buildspec_updates_buildspec_override():
    overrider = prepare_dlc_dev_environment.TomlOverrider()

    valid_buildspec_paths = [
        "pytorch/inference/buildspec-graviton.yml",
        "tensorflow/inference/buildspec-neuronx.yml",
        "huggingface/pytorch/training/buildspec.yml",
    ]

    overrider.set_buildspec(valid_buildspec_paths)

    expected_buildspec_override = {
        "dlc-pr-huggingface-pytorch-training": "huggingface/pytorch/training/buildspec.yml",
        "dlc-pr-pytorch-graviton-inference": "pytorch/inference/buildspec-graviton.yml",
        "dlc-pr-tensorflow-2-neuronx-inference": "tensorflow/inference/buildspec-neuronx.yml",
    }

    assert overrider.overrides["buildspec_override"] == expected_buildspec_override


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("set_buildspec")
@pytest.mark.skipif(
    not is_pr_context(),
    reason="Development environment utility only needs to be tested in PRs, and does not add functional value in other contexts.",
)
def test_set_buildspec_invalid_path():
    overrider = prepare_dlc_dev_environment.TomlOverrider()

    invalid_buildspec_paths = [  # invalid path
        "invalid/path/buildspec.yml",
        "pytorch/invalid/buildspec-aws-graviton2.yml",
        "tensorflow/inference/buildspec-aws-neuronx.yml",
    ]

    with pytest.raises(RuntimeError):
        overrider.set_buildspec(invalid_buildspec_paths)


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("set_buildspec")
@pytest.mark.skipif(
    not is_pr_context(),
    reason="Development environment utility only needs to be tested in PRs, and does not add functional value in other contexts.",
)
def test_set_buildspec_updates_dev_mode():
    overrider = prepare_dlc_dev_environment.TomlOverrider()

    valid_buildspec_paths = [
        "pytorch/inference/buildspec-graviton.yml",
        "tensorflow/inference/buildspec-neuronx.yml",
    ]

    overrider.set_buildspec(valid_buildspec_paths)

    assert overrider.overrides["dev"]["graviton_mode"] is True
    # Only the first dev mode is used, so neuronx is set to False
    assert overrider.overrides["dev"]["neuronx_mode"] is False


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("set_buildspec")
@pytest.mark.skipif(
    not is_pr_context(),
    reason="Development environment utility only needs to be tested in PRs, and does not add functional value in other contexts.",
)
def test_set_buildspec_updates_build_frameworks():
    overrider = prepare_dlc_dev_environment.TomlOverrider()

    valid_buildspec_paths = [
        "pytorch/inference/buildspec-graviton.yml",
        "tensorflow/inference/buildspec-neuronx.yml",
        "huggingface/pytorch/training/buildspec.yml",
    ]

    overrider.set_buildspec(valid_buildspec_paths)

    expected_build_frameworks = ["pytorch", "tensorflow", "huggingface_pytorch"]
    assert overrider.overrides["build"]["build_frameworks"] == expected_build_frameworks


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("set_buildspec")
@pytest.mark.skipif(
    not is_pr_context(),
    reason="Development environment utility only needs to be tested in PRs, and does not add functional value in other contexts.",
)
def test_set_buildspec_updates_build_training_only():
    overrider = prepare_dlc_dev_environment.TomlOverrider()

    buildspec_paths = [
        "pytorch/training/buildspec.yml",
        "huggingface/pytorch/inference/buildspec.yml",
    ]

    overrider.set_buildspec(buildspec_paths)

    assert overrider.overrides["build"]["build_training"] is True
    assert overrider.overrides["build"]["build_inference"] is True


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("set_buildspec")
@pytest.mark.skipif(
    not is_pr_context(),
    reason="Development environment utility only needs to be tested in PRs, and does not add functional value in other contexts.",
)
def test_set_buildspec_updates_build_inference_only():
    overrider = prepare_dlc_dev_environment.TomlOverrider()

    buildspec_paths = [
        "tensorflow/inference/buildspec-neuronx.yml",
    ]

    overrider.set_buildspec(buildspec_paths)

    assert overrider.overrides["build"]["build_training"] is False
    assert overrider.overrides["build"]["build_inference"] is True


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("generate_new_file_content")
@pytest.mark.skipif(
    not is_pr_context(),
    reason="Development environment utility only needs to be tested in PRs, and does not add functional value in other contexts.",
)
def test_generate_new_file_content():
    previous_version_path = "path/to/previous/version/file"
    major_version = "1"
    minor_version = "14"

    mock_file_content = 'version: &VERSION 1.13.0\nshort_version: &SHORT_VERSION "1.13"\n'

    @patch("builtins.open", new_callable=mock_open, read_data=mock_file_content)
    def mock_generate_new_file_content(mock_file):
        expected_content = ["version: &VERSION 1.14.0\n", 'short_version: &SHORT_VERSION "1.14"\n']

        result = prepare_dlc_dev_environment.generate_new_file_content(
            previous_version_path, major_version, minor_version
        )
        assert result == expected_content

    mock_generate_new_file_content()
