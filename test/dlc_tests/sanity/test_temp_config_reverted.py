import pytest
from src.config.test_config import ENABLE_BENCHMARK_DEV_MODE


@pytest.mark.model("N/A")
def test_benchmark_dev_mode_disabled():
    assert not ENABLE_BENCHMARK_DEV_MODE, \
        "Please revert the config change on src/config/test_config.py by setting ENABLE_BENCHMARK_DEV_MODE = False"
