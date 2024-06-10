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
def test_set_test_types_single():
    overrider = prepare_dlc_dev_environment.TomlOverrider()

    for test_type in [
        "benchmark",
        "ec2",
        "ecs",
        "eks",
        "sagemaker_remote",
        "sagemaker_local",
    ]:
        overrider.set_test_types([test_type])
        assert overrider.overrides == {"build": {"test_types": [test_type]}}

    overrider.set_test_types(
        [
            "benchmark",
            "ec2",
            "ecs",
            "eks",
            "sagemaker_remote",
            "sagemaker_local",
        ]
    )
    assert overrider.overrides == {
        "build": {
            "test_types": [
                "benchmark",
                "ec2",
                "ecs",
                "eks",
                "sagemaker_remote",
                "sagemaker_local",
            ]
        }
    }

    overrider.set_test_types(["benchmark", "benchmark", "sagemaker_local"])
    assert overrider.overrides == {"build": {"test_types": ["benchmark", "sagemaker_local"]}}


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("test_types")
def test_set_test_types_empty():  # no tests
    overrider = prepare_dlc_dev_environment.TomlOverrider()
    overrider.set_test_types([])
    assert overrider.overrides == {"build": {"test_types": []}}


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("test_types")
def test_set_test_types_default():  # default tests
    overrider = prepare_dlc_dev_environment.TomlOverrider()
    overrider.set_test_types([])
    assert overrider.overrides == {
        "build": {
            "test_types": [
                "ec2",
                "ecs",
                "eks",
                "sagemaker_remote",
                "sagemaker_local",
            ]
        }
    }


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("dev_mode")
def test_set_dev_mode():
    overrider = prepare_dlc_dev_environment.TomlOverrider()

    overrider.set_dev_mode(None)
    assert overrider.overrides == {
        "dev": {"graviton_mode": False, "neuron_mode": False, "deep_canary_mode": False}
    }

    overrider.set_dev_mode("graviton")
    assert overrider.overrides == {
        "dev": {"graviton_mode": True, "neuron_mode": False, "deep_canary_mode": False}
    }

    overrider.set_dev_mode("neuron")
    assert overrider.overrides == {
        "dev": {"graviton_mode": False, "neuron_mode": True, "deep_canary_mode": False}
    }

    overrider.set_dev_mode("deep_canary")
    assert overrider.overrides == {
        "dev": {"graviton_mode": False, "neuron_mode": False, "deep_canary_mode": True}
    }
