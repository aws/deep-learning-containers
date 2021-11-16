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
from .cpu_allowed_pbtxt_ops import cpu_allowed_pbtxt_ops

from .consts import (
    HBN_TF_GRAPH_DUMP_FLAG,
    TF_DUMP_GRAPH_PREFIX_FLAG,
    CPU,
    HPU_SHORT,
)

#from tests.tf_training_tests.topologies_tests.Utils.library_loader.library_loader import LibraryLoader
#Directly load habana libraries from habana_frameworks
from habana_frameworks.tensorflow.library_loader import load_habana_module

from .Cleaner import Cleaner
from .utils import count_ops_in_graph_dumps

env_flags_backup = {}

def pytest_addoption(parser):
    parser.addoption(
        "--without_hpu", action="store_true", default=False, help="Disable HPU and run tests"
    )
    parser.addoption(
        "--enable_graph_dumps", action="store_true", default=False, help="Enable graph dumps"
    )
    parser.addoption(
        "--run_xfail_only", action="store_true", default=False, help="Runs all the xfail marked tests alone"
    )
    parser.addoption(
        "--run_hang_tests", action="store_true", default=False, help="Runs only the tests which causes hang/crash"
    )
    parser.addoption(
        "--run_jira", action="store", default="", help="Runs the failed tests that are marked under this jira(--run_jira SW-xxxx). To run all the jiras: --run_jira all"
    )

def pytest_sessionstart(session):
    without_hpu = session.config.getoption("--without_hpu")
    if without_hpu:
        return

    # cleanup old pbtxt files
    Cleaner.remove_post_part_files()

    print("########### Trying to load HPU")
    #LibraryLoader.load_habana_libraries()
    load_habana_module()
    print("########### HPU loaded")

    global env_flags_backup
    env_flags_backup[HBN_TF_GRAPH_DUMP_FLAG] = (
        os.environ[HBN_TF_GRAPH_DUMP_FLAG]
        if HBN_TF_GRAPH_DUMP_FLAG in os.environ
        else None
    )
    env_flags_backup[TF_DUMP_GRAPH_PREFIX_FLAG] = (
        os.environ[TF_DUMP_GRAPH_PREFIX_FLAG]
        if TF_DUMP_GRAPH_PREFIX_FLAG in os.environ
        else None
    )

    # set flag to dump graphs
    enable_graph_dumps = session.config.getoption("--enable_graph_dumps")
    if enable_graph_dumps:
        os.environ[HBN_TF_GRAPH_DUMP_FLAG] = "1"
        os.environ[TF_DUMP_GRAPH_PREFIX_FLAG] = "."

def pytest_sessionfinish(session, exitstatus):
    without_hpu = session.config.getoption("--without_hpu")
    if without_hpu:
        return

    enable_graph_dumps = session.config.getoption("--enable_graph_dumps")
    # restore original value of HBN_TF_GRAPH_DUMP_FLAG
    for flag, state in env_flags_backup.items():
        if enable_graph_dumps:
            del os.environ[flag]
        # Restore the original state if not None
        if state is not None:
            os.environ[flag] = state

    if enable_graph_dumps:
        hpu_op_count = count_ops_in_graph_dumps(HPU_SHORT)
        print("\nHPU", hpu_op_count)
        cpu_op_count = count_ops_in_graph_dumps(CPU)
        print("\nCPU", cpu_op_count)
        # if op not listed here will be executed on CPU pytest will fail during teardown
        #ops_executed_on_cpu = set(cpu_op_count.keys())
        #unexpected_ops_executed_on_cpu = ops_executed_on_cpu - cpu_allowed_pbtxt_ops
        # TODO: uncomment, maybe we should check unexpected ops per module or test?
        # if unexpected_ops_executed_on_cpu:
        #     raise RuntimeError(
        #         f"Some ops were executed on CPU, while we expected HPU."
        #         f"List of problematic ops:\n {unexpected_ops_executed_on_cpu}"
        #     )

    Cleaner.remove_post_part_files()
