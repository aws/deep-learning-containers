"""Shared pytest configuration for XGBoost tests."""

import pytest
from test_utils.constants import SAGEMAKER_ROLE


def pytest_addoption(parser):
    # NOTE: skipif/xfail string conditions use lexicographic comparison on this value.
    # This works for single-digit minor versions (3.0.5, 3.2.0, 3.9.x) but would break
    # for 3.10.x+. If that ever happens, switch to a numeric marker or custom plugin.
    parser.addoption(
        "--xgboost-version",
        default="3.2.0",
        help="XGBoost version under test (e.g. 3.0.5 or 3.2.0)",
    )


@pytest.fixture(scope="session")
def xgboost_version(request):
    return request.config.getoption("--xgboost-version")


def _version_tuple(v):
    return tuple(int(x) for x in v.split("."))


@pytest.fixture(scope="session")
def gpu_tree_method(xgboost_version):
    return "gpu_hist" if _version_tuple(xgboost_version) < (3, 2, 0) else "hist"


@pytest.fixture(scope="session")
def role():
    return SAGEMAKER_ROLE
