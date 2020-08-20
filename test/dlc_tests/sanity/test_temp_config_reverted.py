import pytest
from src.config.test_config import ENABLE_BENCHMARK_DEV_MODE


@pytest.mark.model("N/A")
def test_benchmark_dev_mode_disabled():
    assert not ENABLE_BENCHMARK_DEV_MODE
