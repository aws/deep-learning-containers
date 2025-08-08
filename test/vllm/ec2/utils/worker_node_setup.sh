#!/bin/bash

# Usage: ./worker_node_setup.sh <image_uri> <head_ip>
set -e

IMAGE_URI=$1
HEAD_IP=$2

WORKER_IP=$(hostname -i)

tmux new-session -d -s ray_worker "bash vllm/examples/online_serving/run_cluster.sh \
    $IMAGE_URI $HEAD_IP \
    --worker \
    /fsx/.cache/huggingface \
    -e VLLM_HOST_IP=$WORKER_IP \
    -e FI_PROVIDER=efa \
    -e FI_EFA_USE_DEVICE_RDMA=1 \
    --device=/dev/infiniband/ \
    --ulimit memlock=-1:-1
    "

sleep 30

CONTAINER_ID=$(docker ps --filter name=node-* --format "{{.ID}}" | head -n 1)

if [ -z "$CONTAINER_ID" ]; then
    echo "Failed to get container ID"
    exit 1
fi

echo "Worker node container ID: $CONTAINER_ID"
