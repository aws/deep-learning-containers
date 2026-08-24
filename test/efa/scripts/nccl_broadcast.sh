#!/bin/bash
# NCCL broadcast smoke test across 2 nodes over EFA.
set -ex

NUM_HOSTS_FILE=$1
NUM_HOSTS=$2
TIMEOUT_S="${BROADCAST_TIMEOUT_S:-300}"

: "${CUDA_HOME:=/usr/local/cuda}"
export CUDA_HOME

GPU_COUNT=$(nvidia-smi -L | wc -l)
NPROC_PER_NODE=$GPU_COUNT
WORLD_SIZE=$(($GPU_COUNT * $NUM_HOSTS))

MASTER_ADDR=$(head -n1 "$NUM_HOSTS_FILE" | awk '{print $1}')
if [ "$MASTER_ADDR" = "localhost" ]; then
    MASTER_ADDR=$(hostname -I | awk '{print $1}')
fi

BROADCAST_LOG="/test/efa/logs/nccl_broadcast.log"
mkdir -p /test/efa/logs

echo "Running NCCL broadcast test across $NUM_HOSTS nodes ($WORLD_SIZE ranks total)"

set +e
timeout "$TIMEOUT_S" mpirun \
    -x FI_PROVIDER=efa -x FI_EFA_FORK_SAFE=1 -x RDMAV_FORK_SAFE=1 \
    -x NCCL_DEBUG=INFO -x MASTER_ADDR="$MASTER_ADDR" -x MASTER_PORT=29500 \
    -x PATH -x LD_LIBRARY_PATH \
    -n "$WORLD_SIZE" -N "$NPROC_PER_NODE" --hostfile "$NUM_HOSTS_FILE" \
    --mca pml ^cm --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to none \
    bash -c 'RANK=$OMPI_COMM_WORLD_RANK LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK WORLD_SIZE='"$WORLD_SIZE"' python3 /test/efa/scripts/nccl_broadcast.py' \
    2>&1 | tee "$BROADCAST_LOG"
RC=${PIPESTATUS[0]}
set -e

echo "==================== BEGIN ${BROADCAST_LOG} ===================="
cat "${BROADCAST_LOG}" 2>/dev/null || echo "(log missing)"
echo "==================== END ${BROADCAST_LOG} ===================="

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
