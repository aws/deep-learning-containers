#!/bin/bash
set -e

echo "Start NCCL all_reduce local test"
NCCL_ARGS="-b 4 -e 128M -f 2 -g 4"
all_reduce_perf $NCCL_ARGS
echo "Complete NCCL all_reduce local test"
