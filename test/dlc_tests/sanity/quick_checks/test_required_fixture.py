import logging
import os
import sys
import re

import pytest

from test.test_utils import PR_ONLY_REASON, get_repository_local_path, is_pr_context

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stderr))


@pytest.mark.quick_checks
@pytest.mark.skipif(not is_pr_context(), reason=PR_ONLY_REASON)
@pytest.mark.model("N/A")
def test_sanity_fixture():
    """
    Checks that each sanity test that run within PR or MAINLINE contexts
    under test/dlc_tests/sanity/ directory contains either
    `security_sanity` or `functionality_sanity` fixtures, not both.

    This test assumes that each test method declare the fixtures
    using marker pattern `@pytest.mark.usefixtures()` for regex matching.
    """
    repository_path = os.getenv("CODEBUILD_SRC_DIR")
    if not repository_path:
        repository_path = get_repository_local_path()

    # Look only at test files within the sanity test directory
    sanity_test_path = os.path.join(repository_path, "test", "dlc_tests", "sanity")
    LOGGER.debug(f"Test directory: {sanity_test_path}")

    sanity_test_fixture_mapping = {}
    failed_assertion = ""

    # Tests that do not run in PR or MAINLINE contexts do not need to have
    # `security_sanity` or `functionality_sanity` fixtures
    non_pr_mainline_tests = ["test_canary_integration.py::test_deep_canary_integration"]

    # Navigate through files and look at test files at the top level test/dlc_tests/sanity/
    for item in os.listdir(sanity_test_path):
        file_path = os.path.join(sanity_test_path, item)
        if os.path.isfile(file_path):
            _update_test_fixtures_mapping(file_path, sanity_test_fixture_mapping)

    for test_name, test_fixtures in sanity_test_fixture_mapping.items():
        LOGGER.debug(
            f"Checking test method: {test_name} with the following fixtures {test_fixtures}\n"
        )
        # Check only tests that run in PR or MAINLINE contexts
        if test_name not in non_pr_mainline_tests:
            # Append to failed assertion result on XOR condition that the test
            # must have either `security_sanity` or `functionality_sanity` fixture
            if not (
                ("security_sanity" in test_fixtures) ^ ("functionality_sanity" in test_fixtures)
            ):
                failed_assertion = "\n".join(
                    [
                        failed_assertion,
                        f"{test_name} must have either `security_sanity` or `functionality_sanity` fixture, current fixtures: {test_fixtures}",
                    ]
                )

    # Throw assertion error if failed_assertion string is not empty
    assert not failed_assertion, f"{failed_assertion}"


@pytest.mark.quick_checks
@pytest.mark.skipif(not is_pr_context(), reason=PR_ONLY_REASON)
@pytest.mark.model("N/A")
def test_telemetry_fixture():
    """
    Checks that each telemetry test that run within PR or MAINLINE contexts
    under test/dlc_tests/ec2/ directory contains `telemetry` fixture.

    This test assumes that each test method declare the fixtures
    using marker pattern `@pytest.mark.usefixtures()` for regex matching.
    """
    repository_path = os.getenv("CODEBUILD_SRC_DIR")
    if not repository_path:
        repository_path = get_repository_local_path()

    # Look only at ec2 telemetry test file
    telemetry_test_path = os.path.join(
        repository_path, "test", "dlc_tests", "ec2", "test_telemetry.py"
    )
    LOGGER.debug(f"Test path: {telemetry_test_path}")

    telemetry_test_fixture_mapping = {}
    failed_assertion = ""

    # Look at ec2 telemetry test file
    _update_test_fixtures_mapping(telemetry_test_path, telemetry_test_fixture_mapping)

    for test_name, test_fixtures in telemetry_test_fixture_mapping.items():
        LOGGER.debug(
            f"Checking test method: {test_name} with the following fixtures {test_fixtures}\n"
        )
        # Append to failed assertion result if ec2 telemetry tests doesn't contain a `telemetry` fixture
        if "telemetry" not in test_fixtures:
            failed_assertion = "\n".join(
                [
                    failed_assertion,
                    f"{test_name} must have `telemetry` fixture, current fixtures: {test_fixtures}",
                ]
            )

    # Throw assertion error if failed_assertion string is not empty
    assert not failed_assertion, f"{failed_assertion}"


def _update_test_fixtures_mapping(file_to_check, test_fixtures_mapping):
    fixture_pattern = r"@pytest.mark.usefixtures\("
    test_func_pattern = r"def (test_(.*))\("
    fixture_list = []

    with open(file_to_check, "r") as file:
        for line in file:
            # If sees a `usefixtures` marker, add the list of fixtures to fixture_per_test
            # to collect all the fixture names used within a single test method
            if re.match(fixture_pattern, line):
                # If sees a multiline `usefixtures` marker,
                # append line until closing `)` for fixture regex matching
                while ")" not in line:
                    line += next(file)
                # Remove quotes, newlines, tabs, spaces from string
                line = re.sub(r"[\"\n\t\s]*", "", line)
                # Get only the fixture names and remove `@pytest*` prefix
                fixture_regex = re.match(rf"{fixture_pattern}(.*)\)", line)
                fixture_list = fixture_regex.group(1).split(",")
                fixture_list += fixture_list

            # If sees a `test_*` method, update the <test_name> : <fixture_list> dictionary
            if re.match(test_func_pattern, line):
                function_name = re.match(test_func_pattern, line).group(1)
                # Map list of fixtures per tests
                test_fixtures_mapping[
                    f"{os.path.basename(file_to_check)}::{function_name}"
                ] = fixture_list
                # Empty test_fixtures list for the next test method
                fixture_list = []
