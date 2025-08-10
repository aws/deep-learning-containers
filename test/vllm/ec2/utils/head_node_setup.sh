#!/bin/bash

# Usage: ./head_node_setup.sh <image_uri> <hf_token>
set -e

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

IMAGE_URI=$1
HF_TOKEN=$2
HEAD_IP=$(hostname -i)

log "Starting head node setup..."
log "Image URI: $IMAGE_URI"
log "Head IP: $HEAD_IP"

# Start head node in tmux session and capture container ID
tmux new-session -d -s ray_head "bash /fsx/vllm-dlc/vllm/examples/online_serving/run_cluster.sh \
    $IMAGE_URI $HEAD_IP \
    --head \
    /fsx/.cache/huggingface \
    -e VLLM_HOST_IP=$HEAD_IP \
    -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
    -e FI_PROVIDER=efa \
    -e FI_EFA_USE_DEVICE_RDMA=1 \
    --device=/dev/infiniband/ \
    --ulimit memlock=-1:-1 \
    -p 8000:8000"

log "Waiting for container to start..."
sleep 300
log "Head node started"

