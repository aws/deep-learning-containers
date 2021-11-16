###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import os
import sys
import copy
import pytest
from expected_fail_tests import all_xfails_dict, all_hangs_dict, all_cpu_xfails_dict, import_error_files_to_ignore
env_flags_backup = {}

#Ignore all the test files which throw python import errors
collect_ignore = copy.deepcopy(import_error_files_to_ignore)

# Key in the expected_fail_tests can be an exact node_id or module or directory(Ex: tensorflow/python/keras/distribute)
def node_in_test_dict(test_dict, nodeid):
    for key in test_dict:
        if key.endswith('::') or key.endswith('.py') or key.endswith('/'):
            if nodeid.startswith(key):
                return True
        elif key == nodeid:
            return True

    return False

def get_reason_for_node(test_dict, nodeid):
    for key in test_dict:
        if key.endswith('::') or key.endswith('.py') or key.endswith('/'):
            if nodeid.startswith(key):
                return test_dict[key]
        elif key == nodeid:
            return test_dict[key]

    return ""

def pytest_collection_modifyitems(config, items):
    run_xfail_only = config.getoption("--run_xfail_only")
    run_hang_tests_only = config.getoption("--run_hang_tests")
    jira_id = config.getoption("--run_jira")

    deselected_list = []
    if jira_id:
        tests_to_run = []

        if jira_id == "all":
            jira_id = "https://jira.habana-labs.com/browse/"
        else:
            jira_id = "https://jira.habana-labs.com/browse/" + jira_id

        all_fail_tests = {**all_xfails_dict}

        for item in items:
            reason_str = get_reason_for_node(all_fail_tests, item.nodeid)
            if node_in_test_dict(all_fail_tests, item.nodeid) and reason_str.startswith(jira_id):
                xfail_marker = pytest.mark.xfail(run=True, reason=reason_str)
                item.add_marker(xfail_marker)
                tests_to_run.append(item)
            else:
                deselected_list.append(item)

        items[:] = tests_to_run
        config.hook.pytest_deselected(items=deselected_list)
    else:
        xfail_test_list = []
        hang_test_list = []
        for item in items:
            if node_in_test_dict(all_xfails_dict, item.nodeid):
                reason_str=get_reason_for_node(all_xfails_dict, item.nodeid)
                xfail_marker = pytest.mark.xfail(run=run_xfail_only, reason=reason_str)
                xfail_test_list.append(item)
                item.user_properties.append(("xfail", "true"))
                item.add_marker(xfail_marker)
            elif node_in_test_dict(all_hangs_dict, item.nodeid):
                reason_str=get_reason_for_node(all_hangs_dict, item.nodeid)
                xfail_marker = pytest.mark.xfail(run=run_hang_tests_only, reason=reason_str)
                hang_test_list.append(item)
                item.user_properties.append(("xfail", "true"))
                item.add_marker(xfail_marker)
            elif node_in_test_dict(all_cpu_xfails_dict, item.nodeid):
                reason_str=get_reason_for_node(all_cpu_xfails_dict, item.nodeid)
                skip_marker = pytest.mark.skip(reason=reason_str)
                item.add_marker(skip_marker)
            else:
                deselected_list.append(item)

        if run_xfail_only or run_hang_tests_only:
            items.clear()
            items[:] = xfail_test_list + hang_test_list
            config.hook.pytest_deselected(items=deselected_list)
