import json
import logging
import os

logger = logging.getLogger(__name__)


def pytest_collection_modifyitems(config, items):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    marker = config.getoption("-m")

    if marker == "PASSING_TESTS":
        REQ_JSON = os.path.join(base_dir, "habana_tests.json")
    else:
        return

    issue_mapping = {}

    with open(REQ_JSON, "r") as f:
        issue_mapping = json.load(f)

    if marker in issue_mapping.keys():
        all_tests = issue_mapping[marker]
        if len(all_tests) != len(set(issue_mapping[marker])):
            duplicates = set([x for x in all_tests if all_tests.count(x) > 1])
            logger.error(f"Duplicated entries found for marker {marker} : {duplicates}")
            assert 0, f"Duplicated entries found for marker {marker} : {duplicates}"

    else:
        logger.error(f"No testcases marked with marker {marker}")
        return

    collected_tests = []
    for item in items:
        testname = item.name
        if testname in all_tests:
            item.add_marker(marker)
            collected_tests.append(testname)

