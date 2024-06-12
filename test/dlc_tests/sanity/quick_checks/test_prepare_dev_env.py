import pytest

from src import prepare_dlc_dev_environment


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("build_frameworks")
def test_build_frameworks():
    overrider = prepare_dlc_dev_environment.TomlOverrider()
    overrider.set_build_frameworks(("pytorch", "tensorflow"))

    assert overrider.overrides == {"build": {"build_frameworks": ["pytorch", "tensorflow"]}}


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("job_types")
def test_build_job_types():
    overrider = prepare_dlc_dev_environment.TomlOverrider()
    overrider.set_job_type(("inference", "training"))
    assert overrider.overrides == {
        "build": {
            "build_training": True,
            "build_inference": True,
        }
    }

    overrider.set_job_type(["inference"])
    assert overrider.overrides == {
        "build": {
            "build_training": False,
            "build_inference": True,
        }
    }

    overrider.set_job_type(["training"])
    assert overrider.overrides == {
        "build": {
            "build_training": True,
            "build_inference": False,
        }
    }

    overrider.set_job_type([])
    assert overrider.overrides == {
        "build": {
            "build_training": False,
            "build_inference": False,
        }
    }


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("test_types")
def test_set_test_types():
    overrider = prepare_dlc_dev_environment.TomlOverrider()

    # Test case with a subset of test types
    test_types = ["ec2_tests", "ecs_tests", "sagemaker_remote_tests"]
    overrider.set_test_types(test_types)
    expected_overrides = {
        "test": {
            "sanity_tests": True,
            "safety_check_test": False,
            "ecr_scan_allowlist_feature": False,
            "ecs_tests": True,
            "eks_tests": False,
            "ec2_tests": True,
            "ec2_benchmark_tests": False,
            "ec2_tests_on_heavy_instances": False,
            "sagemaker_local_tests": False,
            "sagemaker_remote_tests": True,
            "sagemaker_efa_tests": False,
            "sagemaker_rc_tests": False,
            "sagemaker_benchmark_tests": False,
            "nightly_pr_test_mode": False,
            "use_scheduler": False,
        }
    }
    assert overrider.overrides == expected_overrides

    # Test case with no test types (default behavior)
    test_types = []
    overrider.set_test_types(test_types)
    expected_overrides = {
        "test": {
            "sanity_tests": True,
            "safety_check_test": False,
            "ecr_scan_allowlist_feature": False,
            "ecs_tests": True,
            "eks_tests": True,
            "ec2_tests": True,
            "ec2_benchmark_tests": False,
            "ec2_tests_on_heavy_instances": False,
            "sagemaker_local_tests": True,
            "sagemaker_remote_tests": True,
            "sagemaker_efa_tests": False,
            "sagemaker_rc_tests": False,
            "sagemaker_benchmark_tests": False,
            "nightly_pr_test_mode": False,
            "use_scheduler": False,
        }
    }
    assert overrider.overrides == expected_overrides


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("dev_mode")
def test_set_dev_mode():
    overrider = prepare_dlc_dev_environment.TomlOverrider()

    # test with no dev mode provided
    overrider.set_dev_mode(None)
    assert overrider.overrides == {
        "dev": {"graviton_mode": False, "neuronx_mode": False, "deep_canary_mode": False}
    }

    overrider.set_dev_mode("graviton_mode")
    assert overrider.overrides == {
        "dev": {"graviton_mode": True, "neuronx_mode": False, "deep_canary_mode": False}
    }

    overrider.set_dev_mode("neuronx_mode")
    assert overrider.overrides == {
        "dev": {"graviton_mode": False, "neuronx_mode": True, "deep_canary_mode": False}
    }

    overrider.set_dev_mode("deep_canary_mode")
    assert overrider.overrides == {
        "dev": {"graviton_mode": False, "neuronx_mode": False, "deep_canary_mode": True}
    }

    # Test case with multiple dev modes (error)
    with pytest.raises(ValueError):
        overrider.set_dev_mode(["graviton_mode", "neuronx_mode"])


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("buildspec")
def test_set_buildspec():
    overrider = prepare_dlc_dev_environment.TomlOverrider()

    # Test case: valid buildspec_path
    buildspec_path = "habana/tensorflow/training/buildspec-2-10.yml"
    overrider.set_buildspec(buildspec_path)
    expected_overrides = {
        "buildspec_override": {
            "dlc-pr-tensorflow-2-habana-training": "habana/tensorflow/training/buildspec-2-10.yml"
        }
    }
    assert overrider.overrides == expected_overrides

    # Test case: empty buildspec_path
    buildspec_path = ""
    overrider.set_buildspec(buildspec_path)
    expected_overrides = {"buildspec_override": {}}
    assert overrider.overrides == expected_overrides

    # Test case: invalid buildspec_path format
    buildspec_path = "invalid/path/format.yml"
    with pytest.raises(ValueError):
        overrider.set_buildspec(buildspec_path)
