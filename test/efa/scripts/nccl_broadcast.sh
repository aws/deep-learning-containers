#!/bin/bash
# NCCL broadcast smoke test across 2 nodes over EFA.
set -eux

NUM_HOSTS_FILE=$1
NUM_HOSTS=$2
TIMEOUT_S="${BROADCAST_TIMEOUT_S:-300}"

: "${CUDA_HOME:=/usr/local/cuda}"
export CUDA_HOME

GPU_COUNT=$(nvidia-smi -L | wc -l)
NPROC_PER_NODE=$GPU_COUNT
WORLD_SIZE=$((GPU_COUNT * NUM_HOSTS))

MASTER_ADDR=$(head -n1 "$NUM_HOSTS_FILE" | awk '{print $1}')
if [ "$MASTER_ADDR" = "localhost" ]; then
    MASTER_ADDR=$(hostname -I | awk '{print $1}')
fi

mkdir -p /test/efa/logs
BROADCAST_LOG="/test/efa/logs/nccl_broadcast.log"

echo "==== python3 smoke check ===="
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
echo "==== end smoke check ===="

echo "Running NCCL broadcast test across $NUM_HOSTS nodes ($WORLD_SIZE ranks total)"

set +e
timeout "$TIMEOUT_S" mpirun \
    -x FI_PROVIDER=efa -x FI_EFA_FORK_SAFE=1 -x RDMAV_FORK_SAFE=1 \
    -x NCCL_DEBUG=INFO -x MASTER_ADDR="$MASTER_ADDR" -x MASTER_PORT=29500 \
    -x PYTHONUNBUFFERED=1 \
    -x PATH -x LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}" \
    -x NCCL_SOCKET_IFNAME=^lo \
    -n "$WORLD_SIZE" -N "$NPROC_PER_NODE" --hostfile "$NUM_HOSTS_FILE" \
    --mca pml ^cm --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to none \
    python3 -u /test/efa/scripts/nccl_broadcast.py 2>&1 | tee "$BROADCAST_LOG"
RC=${PIPESTATUS[0]}
set -e

# The pytest wrapper only surfaces this script's output on failure via the
# exception's captured stdout, which is truncated to the TAIL. So emit the
# smallest, highest-signal diagnostics LAST — a full `cat` of the NCCL-INFO
# flooded log would push the real error out of the retained tail.
echo "==================== high-signal broadcast diagnostics ===================="
echo "--- rank starts / warmup / completion / errors (grep) ---"
grep -aE "\[rank |warmup OK|broadcast completed|Traceback|Error|error:|NCCL WARN|ncclInternal|ncclSystem|Aborted|Segmentation|assert|Timeout|out of memory" "$BROADCAST_LOG" 2>/dev/null | tail -n 40 || echo "(no signal lines matched)"
echo "--- last 40 lines of broadcast log ---"
tail -n 40 "$BROADCAST_LOG" 2>/dev/null || echo "(log missing)"
echo "==================== end diagnostics ===================="

if [ "$RC" -eq 124 ]; then
    echo "nccl_broadcast timed out after ${TIMEOUT_S}s"
    exit 1
fi

if [ "$RC" -ne 0 ]; then
    echo "nccl_broadcast failed (rc=$RC)"
    exit 1
fi

grep -q "broadcast completed" "$BROADCAST_LOG" || { echo "no broadcast completion marker"; exit 1; }
echo "nccl_broadcast passed"
