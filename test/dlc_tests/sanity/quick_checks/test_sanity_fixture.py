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
    repository_path = os.getenv("CODEBUILD_SRC_DIR")
    if not repository_path:
        repository_path = get_repository_local_path()

    # Look only at test files within the sanity test directory
    sanity_test_path = os.path.join(repository_path, "test", "dlc_tests", "sanity")
    LOGGER.info(f"Test directory: {sanity_test_path}")

    fixture_pattern = r"@pytest.mark.usefixtures\("
    test_method_pattern = r"def (test_(.*))\("
    fixture_per_test = []

    # Tests that do not in PR or MAINLINE contexts do not need to have
    # `security_sanity` or `functionality_sanity` fixtures
    method_allowlist = [
        "test_deep_canary_integration"
    ]

    # Navigate through files and look at test files at the top level test/dlc_tests/sanity/
    for item in os.listdir(sanity_test_path):
        file_path = os.path.join(sanity_test_path, item)
        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                for line in file:
                    # If sees a `usefixtures` marker, add the list of fixtures to fixture_per_test
                    # to collect all the fixture names used within a single test method
                    if re.match(fixture_pattern, line):
                        # If sees a multiline `usefixtures` marker, append line until see a closing `)`
                        while ")" not in line:
                            line += next(file)
                        line = re.sub(r"[\"\n\t\s]*", "", line)
                        fixture_regex = re.match(rf"{fixture_pattern}(.*)\)", line)
                        pytest_fixtures = fixture_regex.group(1).split(",")
                        fixture_per_test += pytest_fixtures

                    # If sees a `test_*` method, assert XOR condition that the test must have either
                    # `security_sanity` or `functionality_sanity` fixture, not both
                    if re.match(test_method_pattern, line):
                        function_name = re.match(test_method_pattern, line).group(1)
                        LOGGER.info(f"Checking test method: {function_name}\n"
                                    f"with the following fixtures {fixture_per_test}\n"
                                    f"within file: {file_path}")

                        # Don't check tests that do not run in PR or MAINLINE contexts
                        if function_name not in method_allowlist:
                            assert ("security_sanity" in fixture_per_test) ^ (
                                "functionality_sanity" in fixture_per_test
                            ), f"{function_name} must have either `security_sanity` or `functionality_sanity` fixture"

                        # Empty fixture_per_test variable for the next test method
                        fixture_per_test = []
