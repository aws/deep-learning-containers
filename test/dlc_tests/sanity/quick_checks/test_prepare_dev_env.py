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
    overrider.set_test_types(["unit", "integration", "all"])

    # Test with a single test type
    overrider.set_test_types(["unit"])
    assert overrider.overrides == {"build": {"test_types": ["unit"]}}

    overrider.set_test_types(["integration"])
    assert overrider.overrides == {"build": {"test_types": ["integration"]}}

    overrider.set_test_types(["all"])
    assert overrider.overrides == {"build": {"test_types": ["all"]}}

    # Test with multiple test types
    overrider.set_test_types(["unit", "integration", "all"])
    assert overrider.overrides == {"build": {"test_types": ["unit", "integration", "all"]}}

    overrider.set_test_types(["unit", "integration"])
    assert overrider.overrides == {"build": {"test_types": ["unit", "integration"]}}

    overrider.set_test_types(["integration", "all"])
    assert overrider.overrides == {"build": {"test_types": ["integration", "all"]}}

    overrider.set_test_types(["unit", "all"])
    assert overrider.overrides == {"build": {"test_types": ["unit", "all"]}}

    # Test with an empty list
    overrider.set_test_types([])
    assert overrider.overrides == {"build": {"test_types": []}}

    # Test with duplicates
    overrider.set_test_types(["unit", "unit", "integration"])
    assert overrider.overrides == {"build": {"test_types": ["unit", "integration"]}}

    overrider.set_test_types(["unit", "integration", "integration"])
    assert overrider.overrides == {"build": {"test_types": ["unit", "integration"]}}

    overrider.set_test_types(["integration", "all", "all"])
    assert overrider.overrides == {"build": {"test_types": ["integration", "all"]}}

    overrider.set_test_types(["unit", "unit", "integration", "integration", "all", "all"])
    assert overrider.overrides == {"build": {"test_types": ["unit", "integration", "all"]}}

@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("dev_mode")
def test_set_dev_mode():
    overrider = prepare_dlc_dev_environment.TomlOverrider()

    overrider.set_dev_mode(None)
    assert overrider.overrides == {"dev": {"graviton_mode": False, "neuron_mode": False, "deep_canary_mode": False}}

    overrider.set_dev_mode("graviton")
    assert overrider.overrides == {"dev": {"graviton_mode": True, "neuron_mode": False, "deep_canary_mode": False}}

    overrider.set_dev_mode("neuron")
    assert overrider.overrides == {"dev": {"graviton_mode": False, "neuron_mode": True, "deep_canary_mode": False}}

    overrider.set_dev_mode("deep_canary")
    assert overrider.overrides == {"dev": {"graviton_mode": False, "neuron_mode": False, "deep_canary_mode": True}}