import logging
import os
import sys
import re
import glob

import pytest

from pprint import pprint

from test.test_utils import PR_ONLY_REASON, get_repository_local_path, is_pr_context

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stderr))


# @pytest.mark.quick_checks
# @pytest.mark.skipif(not is_pr_context(), reason=PR_ONLY_REASON)
# @pytest.mark.model("N/A")
# def test_sanity_fixture():
#     """
#     Checks that each sanity test that run within PR or MAINLINE contexts
#     under test/dlc_tests/sanity/ directory contains either
#     `security_sanity` or `functionality_sanity`, not both.
#
#     This test assumes that each test method declare the fixtures
#     using marker pattern `@pytest.mark.usefixtures()` for regex matching.
#     """
#     repository_path = os.getenv("CODEBUILD_SRC_DIR")
#     if not repository_path:
#         repository_path = get_repository_local_path()
#
#     # Look only at test files within the sanity test directory
#     sanity_test_path = os.path.join(repository_path, "test", "dlc_tests", "sanity")
#     LOGGER.debug(f"Test directory: {sanity_test_path}")
#
#     fixture_pattern = r"@pytest.mark.usefixtures\("
#     test_method_pattern = r"def (test_(.*))\("
#     test_fixtures = []
#
#     # Tests that do not run in PR or MAINLINE contexts do not need to have
#     # `security_sanity` or `functionality_sanity` fixtures
#     non_pr_mainline_tests = ["test_deep_canary_integration"]
#
#     # Navigate through files and look at test files at the top level test/dlc_tests/sanity/
#     for item in os.listdir(sanity_test_path):
#         file_path = os.path.join(sanity_test_path, item)
#         if os.path.isfile(file_path):
#             with open(file_path, "r") as file:
#                 for line in file:
#                     # If sees a `usefixtures` marker, add the list of fixtures to fixture_per_test
#                     # to collect all the fixture names used within a single test method
#                     if re.match(fixture_pattern, line):
#                         # If sees a multiline `usefixtures` marker,
#                         # append line until closing `)` for fixture regex matching
#                         while ")" not in line:
#                             line += next(file)
#                         # Remove quotes, newlines, tabs, spaces from string
#                         line = re.sub(r"[\"\n\t\s]*", "", line)
#                         # Get only the fixture names and remove `@pytest*` prefix
#                         fixture_regex = re.match(rf"{fixture_pattern}(.*)\)", line)
#                         fixture_list = fixture_regex.group(1).split(",")
#                         test_fixtures += fixture_list
#
#                     # If sees a `test_*` method, look for `security_sanity` or `functionality_sanity`
#                     # in the list of fixtures for that specific test
#                     if re.match(test_method_pattern, line):
#                         function_name = re.match(test_method_pattern, line).group(1)
#                         LOGGER.debug(
#                             f"Checking test method: {function_name}\n"
#                             f"with the following fixtures {test_fixtures}\n"
#                             f"within file: {file_path}"
#                         )
#
#                         # Check only tests that run in PR or MAINLINE contexts
#                         if function_name not in non_pr_mainline_tests:
#                             # Assert XOR condition that the test must have either
#                             # `security_sanity` or `functionality_sanity` fixture
#                             assert ("security_sanity" in test_fixtures) ^ (
#                                 "functionality_sanity" in test_fixtures
#                             ), f"{function_name} must have either `security_sanity` or `functionality_sanity` fixture"
#
#                         # Empty test_fixtures list for the next test method
#                         test_fixtures = []


@pytest.mark.quick_checks
@pytest.mark.skipif(not is_pr_context(), reason=PR_ONLY_REASON)
@pytest.mark.model("N/A")
def test_sanity_fixture_api():
    repository_path = os.getenv("CODEBUILD_SRC_DIR")
    if not repository_path:
        repository_path = get_repository_local_path()

    # Look only at test files within the sanity test directory
    sanity_test_path = os.path.join(repository_path, "test", "dlc_tests", "sanity")
    LOGGER.info(f"Test directory: {sanity_test_path}")

    files = glob.glob("test_*.py", root_dir=sanity_test_path)

    for file in files:
        nodes = pytest.main([str(file), "--collect-only"])
        LOGGER.info(nodes)
