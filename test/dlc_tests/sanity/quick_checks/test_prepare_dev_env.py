import pytest

from src import prepare_dlc_dev_environment


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("build_frameworks")
def test_build_frameworks():
    assert prepare_dlc_dev_environment.set_build_frameworks(("pytorch", "tensorflow")) == {"build_frameworks": ["pytorch", "tensorflow"]}
