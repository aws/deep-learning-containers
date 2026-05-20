#!/bin/bash
# Verify the EFA test has a working all_reduce implementation.
#
# - PyTorch DLCs bake nccl-tests' all_reduce_perf at build time → preferred.
# - vLLM Ubuntu doesn't bake the binary, and building nccl-tests at test time
#   OOM-kills nvcc on the test host (verifiable.cu is template-heavy and peaks
#   at >8GB). Instead, nccl_allreduce.sh falls back to torch.distributed which
#   exercises the same NCCL→aws-ofi-nccl→EFA path without any compilation.
#
# Both paths are validated by nccl_allreduce.sh; this script just sanity-checks
# that one of them is available.
set -ex

if [ -x /usr/local/bin/all_reduce_perf ]; then
    echo "all_reduce_perf preinstalled at /usr/local/bin/all_reduce_perf"
    exit 0
fi

if python3 -c "import torch.distributed" >/dev/null 2>&1 \
    && [ -f /test/efa/scripts/torch_allreduce.py ]; then
    echo "torch.distributed all_reduce path available"
    exit 0
fi

echo "ERROR: neither all_reduce_perf nor torch.distributed is available"
exit 1
