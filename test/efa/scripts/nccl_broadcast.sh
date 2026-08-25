#!/bin/bash
# NCCL broadcast smoke test across 2 nodes over EFA.
#
# Always exits 0 and prints a final "NCCL_BROADCAST_RESULT: PASS|FAIL" marker.
# The caller (test_efa.py) asserts on that marker. This keeps the full mpirun
# output on the success path (printed verbatim, not truncated by pytest's
# exception formatting), so a failing rank's traceback is always visible.
set -x

NUM_HOSTS_FILE=$1
NUM_HOSTS=$2
TIMEOUT_S="${BROADCAST_TIMEOUT_S:-300}"

: "${CUDA_HOME:=/usr/local/cuda}"
export CUDA_HOME

GPU_COUNT=$(nvidia-smi -L | wc -l)
NPROC_PER_NODE=$GPU_COUNT
WORLD_SIZE=$((GPU_COUNT * NUM_HOSTS))

MASTER_ADDR=$(head -n1 "$NUM_HOSTS_FILE" | awk '{print $1}')
if [ "$MASTER_ADDR" = "localhost" ] || [ -z "$MASTER_ADDR" ]; then
    # `hostname` isn't installed in the DLC image; derive the primary private
    # IP via a routing-table lookup (no packets sent, no DNS needed). This is
    # the address workers use to reach rank 0's c10d TCP store.
    MASTER_ADDR=$(python3 -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(s.getsockname()[0]); s.close()")
fi

mkdir -p /test/efa/logs
BROADCAST_LOG="/test/efa/logs/nccl_broadcast.log"

echo "==== python3 smoke check ===="
python3 -c "import torch; print('torch', torch.__version__, 'cuda_avail', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"
echo "==== end smoke check ===="

echo "Running NCCL broadcast test across $NUM_HOSTS nodes ($WORLD_SIZE ranks total), master=$MASTER_ADDR"

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

echo "==================== broadcast result ===================="
if [ "$RC" -eq 124 ]; then
    echo "NCCL_BROADCAST_RESULT: FAIL (timed out after ${TIMEOUT_S}s)"
elif [ "$RC" -ne 0 ]; then
    echo "NCCL_BROADCAST_RESULT: FAIL (mpirun rc=$RC)"
elif ! grep -q "broadcast completed" "$BROADCAST_LOG"; then
    echo "NCCL_BROADCAST_RESULT: FAIL (no completion marker in output)"
else
    echo "NCCL_BROADCAST_RESULT: PASS"
fi
echo "=========================================================="
exit 0
