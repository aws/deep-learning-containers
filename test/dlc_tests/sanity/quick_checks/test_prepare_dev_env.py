import pytest

from src import prepare_dlc_dev_environment
from unittest.mock import patch


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("build_frameworks")
def test_build_frameworks():
    overrider = prepare_dlc_dev_environment.TomlOverrider()
    overrider.set_build_frameworks(("pytorch", "tensorflow"))

    assert overrider.overrides["build"]["build_frameworks"] == ["pytorch", "tensorflow"]


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("job_types")
def test_build_job_types():
    overrider = prepare_dlc_dev_environment.TomlOverrider()
    overrider.set_job_type(("inference", "training"))
    assert (
        overrider.overrides["build"]["build_training"] == True
        and overrider.overrides["build"]["build_inference"] == True
    )

    overrider.set_job_type(["inference"])
    assert (
        overrider.overrides["build"]["build_training"] == False
        and overrider.overrides["build"]["build_inference"] == True
    )

    overrider.set_job_type(["training"])
    assert (
        overrider.overrides["build"]["build_training"] == True
        and overrider.overrides["build"]["build_inference"] == False
    )

    overrider.set_job_type([])
    assert (
        overrider.overrides["build"]["build_training"] == False
        and overrider.overrides["build"]["build_inference"] == False
    )


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("test_types")
def test_set_test_types():
    overrider = prepare_dlc_dev_environment.TomlOverrider()

    # Test case with a subset of test types
    test_types = ["ec2_tests", "ecs_tests", "sagemaker_remote_tests"]
    overrider.set_test_types(test_types)
    assert overrider.overrides["test"]["sanity_tests"] == False
    assert overrider.overrides["test"]["ecs_tests"] == True
    assert overrider.overrides["test"]["eks_tests"] == False
    assert overrider.overrides["test"]["ec2_tests"] == True
    assert overrider.overrides["test"]["sagemaker_local_tests"] == False
    assert overrider.overrides["test"]["sagemaker_remote_tests"] == True

    # Test case with no test types (default behavior)
    test_types = []
    overrider.set_test_types(test_types)
    assert overrider.overrides["test"]["sanity_tests"] == True
    assert overrider.overrides["test"]["ecs_tests"] == True
    assert overrider.overrides["test"]["eks_tests"] == True
    assert overrider.overrides["test"]["ec2_tests"] == True
    assert overrider.overrides["test"]["sagemaker_local_tests"] == True
    assert overrider.overrides["test"]["sagemaker_remote_tests"] == True


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("dev_mode")
def test_set_dev_mode():
    overrider = prepare_dlc_dev_environment.TomlOverrider()

    # test with no dev mode provided
    overrider.set_dev_mode(None)
    assert overrider.overrides["dev"]["graviton_mode"] == False
    assert overrider.overrides["dev"]["neuronx_mode"] == False
    assert overrider.overrides["dev"]["deep_canary_mode"] == False

    overrider.set_dev_mode("graviton_mode")
    assert overrider.overrides["dev"]["graviton_mode"] == True
    assert overrider.overrides["dev"]["neuronx_mode"] == False
    assert overrider.overrides["dev"]["deep_canary_mode"] == False

    overrider.set_dev_mode("neuronx_mode")
    assert overrider.overrides["dev"]["graviton_mode"] == False
    assert overrider.overrides["dev"]["neuronx_mode"] == True
    assert overrider.overrides["dev"]["deep_canary_mode"] == False

    overrider.set_dev_mode("deep_canary_mode")
    assert overrider.overrides["dev"]["graviton_mode"] == False
    assert overrider.overrides["dev"]["neuronx_mode"] == False
    assert overrider.overrides["dev"]["deep_canary_mode"] == True

    # Test case with multiple dev modes (error)
    with pytest.raises(ValueError):
        overrider.set_dev_mode(["graviton_mode", "neuronx_mode"])


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("set_buildspec")
def test_set_buildspec_updates_buildspec_override():
    overrider = prepare_dlc_dev_environment.TomlOverrider()

    valid_buildspec_paths = [
        "pytorch/training/buildspec-aws-graviton2.yml",
        "tensorflow/inference/buildspec-aws-neuronx-py38.yml",
        "huggingface/training/buildspec-aws-depcanary.yml",
    ]

    overrider.set_buildspec(valid_buildspec_paths)

    expected_buildspec_override = {
        "dlc-pr-pytorch-training": "pytorch/training/buildspec-aws-graviton2.yml",
        "dlc-pr-tensorflow-2-neuronx-inference": "tensorflow/inference/buildspec-aws-neuronx-py38.yml",
        "dlc-pr-huggingface-depcanary-training": "huggingface/training/buildspec-aws-depcanary.yml",
    }

    assert overrider.overrides["buildspec_override"] == expected_buildspec_override


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("set_buildspec")
def test_set_buildspec_invalid_path():
    overrider = prepare_dlc_dev_environment.TomlOverrider()

    invalid_buildspec_paths = [  # invalid path
        "invalid/path/buildspec.yml",
        "pytorch/invalid/buildspec-aws-graviton2.yml",
        "tensorflow/inference/buildspec-aws-neuronx.yml",
    ]

    with pytest.raises(ValueError):
        overrider.set_buildspec(invalid_buildspec_paths)


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("set_buildspec")
def test_set_buildspec_updates_dev_mode():
    overrider = prepare_dlc_dev_environment.TomlOverrider()

    valid_buildspec_paths = [
        "pytorch/training/buildspec-aws-graviton2.yml",
        "tensorflow/inference/buildspec-aws-neuronx-py38.yml",
        "huggingface/training/buildspec-aws-depcanary.yml",
    ]

    overrider.set_buildspec(valid_buildspec_paths)

    assert overrider.overrides["dev"]["graviton_mode"] == True
    assert overrider.overrides["dev"]["neuronx_mode"] == True
    assert overrider.overrides["dev"]["deep_canary_mode"] == True


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("set_buildspec")
def test_set_buildspec_updates_build_frameworks():
    overrider = prepare_dlc_dev_environment.TomlOverrider()

    valid_buildspec_paths = [
        "pytorch/training/buildspec-aws-graviton2.yml",
        "tensorflow/inference/buildspec-aws-neuronx-py38.yml",
        "huggingface/training/buildspec-aws-depcanary.yml",
    ]

    overrider.set_buildspec(valid_buildspec_paths)

    expected_build_frameworks = ["pytorch", "tensorflow", "huggingface"]
    assert overrider.overrides["build"]["build_frameworks"] == expected_build_frameworks


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("set_buildspec")
def test_set_buildspec_updates_build_training_only():
    overrider = prepare_dlc_dev_environment.TomlOverrider()

    buildspec_paths = [
        "pytorch/training/buildspec-aws-graviton2.yml",
        "huggingface/training/buildspec-aws-depcanary.yml",
    ]

    overrider.set_buildspec(buildspec_paths)

    assert overrider.overrides["build"]["build_training"] == True
    assert overrider.overrides["build"]["build_inference"] == False


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("set_buildspec")
def test_set_buildspec_updates_build_inference_only():
    overrider = prepare_dlc_dev_environment.TomlOverrider()

    buildspec_paths = [
        "tensorflow/inference/buildspec-aws-neuronx-py38.yml",
    ]

    overrider.set_buildspec(buildspec_paths)

    assert overrider.overrides["build"]["build_training"] == False
    assert overrider.overrides["build"]["build_inference"] == True
