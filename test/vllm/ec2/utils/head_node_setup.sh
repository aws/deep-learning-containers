#!/bin/bash

# Usage: ./head_node_setup.sh <image_uri> <hf_token> <head_ip>
set -e

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

IMAGE_URI=$1
HF_TOKEN=$2
HEAD_IP=$3
CONTAINER_NAME=$4

log "Starting head node setup..."
log "Image URI: $IMAGE_URI"
log "Head IP: $HEAD_IP"

# Start head node in tmux session and capture container ID
tmux new-session -d -s ray_head "docker run \
    --name $CONTAINER_NAME \
    --network host \
    --shm-size 10.24g \
    --gpus all \
    -v /fsx/.cache/huggingface:/root/.cache/huggingface \
    -e VLLM_HOST_IP=$HEAD_IP \
    -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
    -e FI_PROVIDER=efa \
    -e FI_EFA_USE_DEVICE_RDMA=1 \
    --device=/dev/infiniband/ \
    --ulimit memlock=-1:-1 \
    -p 8000:8000 \
    $IMAGE_URI \
    /bin/bash -c 'ray start --head --block --port=6379'"

log "Head node started in container: $CONTAINER_NAME"

