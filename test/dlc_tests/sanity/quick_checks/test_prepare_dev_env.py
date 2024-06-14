import pytest

from src import prepare_dlc_dev_environment


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


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("test_types")
def test_set_test_types_default():
    overrider = prepare_dlc_dev_environment.TomlOverrider()

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
