import pytest

from src import prepare_dlc_dev_environment


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("build_frameworks")
def test_build_frameworks():
    overrider = prepare_dlc_dev_environment.TomlOverrider()
    overrider.set_build_frameworks(("pytorch", "tensorflow"))

    assert overrider.overrides == {"build":{"build_frameworks": ["pytorch", "tensorflow"]}}


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
