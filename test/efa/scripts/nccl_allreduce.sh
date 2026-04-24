#!/bin/bash
# Ported from V1: test/v2/ec2/efa/testEFA
# Run all_reduce_perf across multiple nodes and verify EFA transport.
set -ex

NUM_HOSTS_FILE=$1
NUM_HOSTS=$2

if [[ -z "${CUDA_HOME}" ]]; then
    echo "CUDA_HOME variable is empty, please define it in dockerfile"
    exit 1
fi

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
    # Debug: show log file size and full content when parse fails
    echo "=== Log file info ==="
    ls -la ${TRAINING_LOG}
    echo "=== Log wc -l ==="
    wc -l ${TRAINING_LOG}
    echo "=== Full log content ==="
    cat ${TRAINING_LOG}
    echo "=== End full log ==="

    # Match data rows only: start with optional whitespace then the byte size
    # Extract out-of-place busbw (column 11 in nccl-tests output)
    benchmark=$(grep -E '^\s*1073741824\s' ${TRAINING_LOG} | tail -n1 | awk '{print $11}')
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

echo "Running all_reduce_perf test"
mpirun -x FI_PROVIDER="efa" -n $NODES -N $GPU_COUNT --hostfile $NUM_HOSTS_FILE \
    -x NCCL_DEBUG=INFO ${USE_DEVICE_RDMA_ARG} -x NCCL_PROTO=simple -x NCCL_ALGO=ring -x RDMAV_FORK_SAFE=1 \
    -x PATH -x LD_LIBRARY_PATH=${CUDA_HOME}/lib:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH \
    -x NCCL_SOCKET_IFNAME=^lo --mca pml ^cm --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to none \
    /all_reduce_perf -b 8 -e 1G -f 2 -g 1 -c 1 -n 100 2>&1 | tee "${TRAINING_LOG}"

RETURN_VAL=${PIPESTATUS[0]}
if [ ${RETURN_VAL} -ne 0 ]; then
    echo "check_efa_nccl_all_reduce failed — mpirun exited ${RETURN_VAL}"
    echo "=== Full log content ==="
    cat ${TRAINING_LOG}
    echo "=== End full log ==="
    exit 1
fi
echo "check_efa_nccl_all_reduce passed"

validate_all_reduce_performance_logs
check_efa_nccl_all_reduce_performance
