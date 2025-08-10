#!/bin/bash

# Usage: ./worker_node_setup.sh <image_uri> <head_ip>
set -e
LOG_FILE="/fsx/vllm-dlc/worker_node_setup.log"


log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

IMAGE_URI=$1
HEAD_IP=$2

WORKER_IP=$(hostname -i)

bash /fsx/vllm-dlc/vllm/examples/online_serving/run_cluster.sh \
    $IMAGE_URI \
    $HEAD_IP \
    --worker \
    /fsx/.cache/huggingface \
    -e VLLM_HOST_IP=$WORKER_IP \
    -e FI_PROVIDER=efa \
    -e FI_EFA_USE_DEVICE_RDMA=1 \
    --device=/dev/infiniband/ \
    --ulimit memlock=-1:-1

log "Worker node setup complete."