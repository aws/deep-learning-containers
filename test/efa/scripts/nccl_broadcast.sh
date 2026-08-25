#!/bin/bash
# NCCL broadcast smoke test across 2 nodes over EFA, launched with torchrun.
#
# Runs one torchrun agent per node: call with node_rank 0 on the master and
# node_rank 1 on the worker, concurrently. Rank 0 hosts the rendezvous on
# master_addr:master_port; the worker connects to it.
#
# Always exits 0 and prints a final "NCCL_BROADCAST_RESULT: PASS|FAIL" marker.
# The caller (test_efa.py) asserts on the master node's marker, so the full
# torchrun output stays visible (not truncated by pytest's exception
# formatting) and a failing rank's traceback is always readable.
set -x

NODE_RANK=$1
MASTER_ADDR=$2
NNODES="${3:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"
TIMEOUT_S="${BROADCAST_TIMEOUT_S:-300}"

: "${CUDA_HOME:=/usr/local/cuda}"
export CUDA_HOME
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1
export RDMAV_FORK_SAFE=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo
export PYTHONUNBUFFERED=1

NPROC_PER_NODE=$(nvidia-smi -L | wc -l)

mkdir -p /test/efa/logs
BROADCAST_LOG="/test/efa/logs/nccl_broadcast_node${NODE_RANK}.log"

echo "==== python3 smoke check ===="
python3 -c "import torch; print('torch', torch.__version__, 'cuda_avail', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"
echo "==== end smoke check ===="

echo "Running NCCL broadcast test (torchrun): node_rank=$NODE_RANK nnodes=$NNODES nproc_per_node=$NPROC_PER_NODE master=$MASTER_ADDR:$MASTER_PORT"

timeout "$TIMEOUT_S" torchrun \
    --nnodes="$NNODES" --nproc_per_node="$NPROC_PER_NODE" --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    /test/efa/scripts/nccl_broadcast.py 2>&1 | tee "$BROADCAST_LOG"
RC=${PIPESTATUS[0]}

echo "==================== broadcast result (node $NODE_RANK) ===================="
if [ "$RC" -eq 124 ]; then
    echo "NCCL_BROADCAST_RESULT: FAIL (timed out after ${TIMEOUT_S}s)"
elif [ "$RC" -ne 0 ]; then
    echo "NCCL_BROADCAST_RESULT: FAIL (torchrun rc=$RC)"
elif [ "$NODE_RANK" -eq 0 ] && ! grep -q "broadcast completed" "$BROADCAST_LOG"; then
    # Rank 0 lives on node_rank 0, so its completion line only appears here.
    echo "NCCL_BROADCAST_RESULT: FAIL (no completion marker in output)"
else
    echo "NCCL_BROADCAST_RESULT: PASS"
fi
echo "=========================================================="
exit 0
