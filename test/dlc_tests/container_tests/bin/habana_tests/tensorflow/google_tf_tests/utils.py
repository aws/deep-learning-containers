from .consts import (
    HPU_CLUSTER_DUMP_FILES,
    POST_PART_DUMP_FILES,
    HPU_SHORT,
)
import glob
import tensorflow as tf
import os
from collections import defaultdict
from google.protobuf import text_format


def count_ops_in_graph_dumps(device_name):
    op_count = defaultdict(lambda: 0)

    # CPU will find all relevant ops in these files.
    # For HPU it will be mostly HabanaLaunch ops, but there can be also ops like
    # shape, size and similar.
    files = glob.glob(POST_PART_DUMP_FILES)
    # TODO: perform some experiment for: shape, size, range and similar ops.
    # Create new generic_multi_variant test for them and and check performance of whole solution.
    # This specific test case should check *only* POST_PART_DUMP_FILES for CPU and HPU while
    # all other tests should check *only* POST_PART_DUMP_FILES for CPU and HPU_CLUSTER_DUMP_FILES for HPU

    if device_name == HPU_SHORT:
        # for regular compute nodes we need to lock inside files which define what
        # HabanaLaunch actually is
        files += glob.glob(HPU_CLUSTER_DUMP_FILES)

    for f in files:
        with open(f, "r") as pbtxt:
            pbtxt_content = pbtxt.read()
            graph_def = text_format.Parse(pbtxt_content, tf.compat.v1.GraphDef())
            op_count = _count_ops_in_graph_nodes(device_name, graph_def, op_count)

    return op_count


def _count_ops_in_graph_nodes(device_name, graph_def, in_out_op_count):
    for node in graph_def.node:
        if device_name in node.device:
            in_out_op_count[node.op] += 1
    return in_out_op_count


def get_sys_variables_as_string():
    sys_var = "System variables:\n"
    for var, val in os.environ.items():
        sys_var = sys_var + f'{var} = {val}\n'
    return sys_var
