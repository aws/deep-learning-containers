#!/bin/bash
# Ported from V1: test/v2/ec2/efa/testEFA
# Run all_reduce_perf across multiple nodes and verify EFA transport.
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

TRAINING_LOG="/test/efa/logs/testEFA.log"
mkdir -p /test/efa/logs

USE_DEVICE_RDMA_ARG=""
if [[ ${INSTANCE_TYPE} == p4d.24xlarge || ${INSTANCE_TYPE} == p4de.24xlarge || ${INSTANCE_TYPE} == p5.48xlarge ]]; then
    USE_DEVICE_RDMA_ARG="-x FI_EFA_USE_DEVICE_RDMA=1"
fi

validate_all_reduce_performance_logs(){
    grep "aws-ofi-nccl" ${TRAINING_LOG} || { echo "aws-ofi-nccl is not working"; exit 1; }
    grep -i "NET/OFI Selected provider is efa" ${TRAINING_LOG} || { echo "EFA provider not selected"; exit 1; }
    grep -E "Using network (AWS )?Libfabric" ${TRAINING_LOG} || { echo "Libfabric not active"; exit 1; }
    if [[ ${INSTANCE_TYPE} == p4d* || ${INSTANCE_TYPE} == p5* ]]; then
        grep "NCCL_TOPO_FILE set by environment to" ${TRAINING_LOG}
        grep -E "NET/(AWS )?Libfabric/0/GDRDMA" ${TRAINING_LOG}
    fi
}

check_efa_nccl_all_reduce_performance(){
    # Match V1: col 11 on the 1 GiB row (in-place algbw).
    benchmark=$(cat $TRAINING_LOG | grep '1073741824' | tail -n1 | awk -F " " '{print $11}' | sed 's/ //' | sed 's/  5e-07//')
    echo "Benchmark throughput: ${benchmark}"
    if [[ -z "${benchmark}" ]]; then
        echo "benchmark variable is empty"
        exit 1
    fi
    PERFORMANCE_THRESHOLD="3"
    if [[ $(echo "$benchmark $PERFORMANCE_THRESHOLD" | awk '{print ($1 >= $2)}') == 1 ]]; then
        echo "check_efa_nccl_all_reduce_performance passed"
    else
        echo "check_efa_nccl_all_reduce_performance failed"
        exit 1
    fi
}

# Capture diagnostics to a file we cat at the very end. invoke/Fabric truncate
# the .stdout of a failing remote command to the last few KB, so anything
# printed before mpirun gets dropped. Stage it through a file and dump after
# the validators run.
DIAG_LOG="/test/efa/logs/diagnostics.log"
{
    echo "==================== EFA / NCCL diagnostics ===================="
    echo "--- nvidia-smi ---"
    nvidia-smi -L || true
    echo "--- libnccl resolution ---"
    ldconfig -p | grep libnccl || echo "(no libnccl in ldconfig)"
    echo "--- ldd all_reduce_perf ---"
    ldd /usr/local/bin/all_reduce_perf 2>&1 | grep -E "nccl|cuda|fabric|not found" || true
    echo "--- libfabric provider list ---"
    fi_info -p efa 2>&1 | head -20 || true
    echo "--- aws-ofi-nccl plugin ---"
    ls -la /opt/amazon/ofi-nccl/lib*/libnccl-net*.so 2>&1 | head -5 || true
    echo "--- /etc/ld.so.conf.d ---"
    ls /etc/ld.so.conf.d/ 2>&1
    echo "==================== end diagnostics ===================="
} > "${DIAG_LOG}" 2>&1

echo "Running all_reduce_perf test"
mpirun -x FI_PROVIDER="efa" -x FI_EFA_FORK_SAFE=1 -n $NODES -N $GPU_COUNT --hostfile $NUM_HOSTS_FILE \
    -x NCCL_DEBUG=INFO ${USE_DEVICE_RDMA_ARG} -x NCCL_PROTO=simple -x NCCL_ALGO=ring -x RDMAV_FORK_SAFE=1 \
    -x PATH -x LD_LIBRARY_PATH=${CUDA_HOME}/lib:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH \
    -x NCCL_SOCKET_IFNAME=^lo --mca pml ^cm --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to none \
    /usr/local/bin/all_reduce_perf -b 8 -e 1G -f 2 -g 1 -c 1 -n 100 2>&1 | tee "${TRAINING_LOG}"

RETURN_VAL=${PIPESTATUS[0]}
if [ ${RETURN_VAL} -eq 0 ]; then
    echo "check_efa_nccl_all_reduce passed"
else
    echo "check_efa_nccl_all_reduce failed"
fi

# Always dump the full mpirun + NCCL_DEBUG log so a failing run has actionable
# evidence in pytest's captured output (the file lives only in the master
# container which gets torn down on test exit). Diagnostics file too — it
# was written before mpirun and would have been truncated by Fabric otherwise.
echo "==================== BEGIN ${DIAG_LOG} ===================="
cat "${DIAG_LOG}" 2>/dev/null || echo "(diagnostics file missing)"
echo "==================== END ${DIAG_LOG} ===================="

echo "==================== BEGIN ${TRAINING_LOG} ===================="
cat "${TRAINING_LOG}" 2>/dev/null || echo "(log file missing)"
echo "==================== END ${TRAINING_LOG} ===================="

validate_all_reduce_performance_logs
check_efa_nccl_all_reduce_performance
