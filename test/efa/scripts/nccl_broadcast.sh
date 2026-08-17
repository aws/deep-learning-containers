#!/bin/bash
# Run broadcast_perf across multiple nodes and verify EFA transport + completion.
#
# On a working image broadcast_perf completes the size sweep and
# prints its "Avg bus bandwidth" summary line, which check_broadcast_completed
# asserts on.
set -ex

NUM_HOSTS_FILE=$1
NUM_HOSTS=$2

# Default CUDA_HOME for images that don't export it (vLLM Ubuntu).
# PyTorch DLCs already set this in the Dockerfile so this is a no-op there.
: "${CUDA_HOME:=/usr/local/cuda}"
export CUDA_HOME

TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_TYPE=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -v http://169.254.169.254/latest/meta-data/instance-type)

GPU_COUNT=$(nvidia-smi -L | wc -l)
NODES=$(($GPU_COUNT * $NUM_HOSTS))

TRAINING_LOG="/test/efa/logs/testEFABroadcast.log"
mkdir -p /test/efa/logs

USE_DEVICE_RDMA_ARG=""
if [[ ${INSTANCE_TYPE} == p4d.24xlarge || ${INSTANCE_TYPE} == p4de.24xlarge || ${INSTANCE_TYPE} == p5.48xlarge ]]; then
    USE_DEVICE_RDMA_ARG="-x FI_EFA_USE_DEVICE_RDMA=1"
fi

validate_broadcast_transport_logs(){
    grep "aws-ofi-nccl" ${TRAINING_LOG} || { echo "aws-ofi-nccl is not working"; exit 1; }
    grep -i "NET/OFI Selected provider is efa" ${TRAINING_LOG} || { echo "EFA provider not selected"; exit 1; }
    grep -E "Using network (AWS )?Libfabric" ${TRAINING_LOG} || { echo "Libfabric not active"; exit 1; }
    if [[ ${INSTANCE_TYPE} == p4d* || ${INSTANCE_TYPE} == p5* ]]; then
        grep "NCCL_TOPO_FILE set by environment to" ${TRAINING_LOG}
        grep -E "NET/(AWS )?Libfabric/0/GDRDMA" ${TRAINING_LOG}
    fi
}

check_broadcast_completed(){
    # nccl-tests prints this summary line only after the full size sweep
    # finishes. If broadcast hangs (the aws-ofi-nccl 1.18.0 bug) the line never
    # appears; the calling test step times out rather than reaching here.
    grep -E "# Avg bus bandwidth" ${TRAINING_LOG} || \
        { echo "broadcast did not complete (no summary line) - possible hang"; exit 1; }
    echo "check_broadcast_completed passed"
}

# Capture diagnostics to a file we cat at the very end. invoke/Fabric truncate
# the .stdout of a failing remote command to the last few KB, so anything
# printed before mpirun gets dropped. Stage it through a file and dump after
# the validators run.
DIAG_LOG="/test/efa/logs/diagnostics_broadcast.log"
{
    echo "==================== EFA / NCCL diagnostics ===================="
    echo "--- nvidia-smi ---"
    nvidia-smi -L || true
    echo "--- libnccl resolution ---"
    ldconfig -p | grep libnccl || echo "(no libnccl in ldconfig)"
    echo "--- ldd broadcast_perf ---"
    ldd /usr/local/bin/broadcast_perf 2>&1 | grep -E "nccl|cuda|fabric|not found" || true
    echo "--- libfabric provider list ---"
    fi_info -p efa 2>&1 | head -20 || true
    echo "--- aws-ofi-nccl plugin ---"
    ls -la /opt/amazon/ofi-nccl/lib*/libnccl-net*.so 2>&1 | head -5 || true
    echo "==================== end diagnostics ===================="
} > "${DIAG_LOG}" 2>&1

echo "Running broadcast_perf test"
mpirun -x FI_PROVIDER="efa" -x FI_EFA_FORK_SAFE=1 -n $NODES -N $GPU_COUNT --hostfile $NUM_HOSTS_FILE \
    -x NCCL_DEBUG=INFO ${USE_DEVICE_RDMA_ARG} -x NCCL_PROTO=simple -x NCCL_ALGO=ring -x RDMAV_FORK_SAFE=1 \
    -x PATH -x LD_LIBRARY_PATH=${CUDA_HOME}/lib:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH \
    -x NCCL_SOCKET_IFNAME=^lo --mca pml ^cm --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to none \
    /usr/local/bin/broadcast_perf -b 8 -e 1G -f 2 -g 1 -c 1 -n 100 2>&1 | tee "${TRAINING_LOG}"

RETURN_VAL=${PIPESTATUS[0]}
if [ ${RETURN_VAL} -eq 0 ]; then
    echo "check_efa_nccl_broadcast passed"
else
    echo "check_efa_nccl_broadcast failed"
fi

# Dump training log first, then the most actionable diagnostics LAST so they
# survive Fabric's stdout truncation.
echo "==================== BEGIN ${TRAINING_LOG} ===================="
cat "${TRAINING_LOG}" 2>/dev/null || echo "(log file missing)"
echo "==================== END ${TRAINING_LOG} ===================="

echo "==================== BEGIN ${DIAG_LOG} ===================="
cat "${DIAG_LOG}" 2>/dev/null || echo "(diagnostics file missing)"
echo "==================== END ${DIAG_LOG} ===================="

validate_broadcast_transport_logs
check_broadcast_completed
