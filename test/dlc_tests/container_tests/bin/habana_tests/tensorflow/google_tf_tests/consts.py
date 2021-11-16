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

HBN_TF_GRAPH_DUMP_FLAG = 'HBN_TF_GRAPH_DUMP'
TF_DUMP_GRAPH_PREFIX_FLAG = "TF_DUMP_GRAPH_PREFIX"
CPU = 'CPU'
HPU_SHORT = 'HPU'

POST_PART_FILES_TO_REMOVE = ["./*PostPartitioning*", "./habana*.pb", "./*.pbtxt"]
ALL_GRAPHS_DUMPS = ".graph_dumps/*"

HPU_CLUSTER_DUMP_FILES = "habana_segment_fdef_graph_habana_cluster*.pbtxt"
POST_PART_DUMP_FILES = "save_pass_PostPartitioning*.pbtxt"
