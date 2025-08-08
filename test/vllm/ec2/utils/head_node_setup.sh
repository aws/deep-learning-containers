#!/bin/bash

# Usage: ./head_node_setup.sh <image_uri> <hf_token>
set -e

IMAGE_URI=$1
HF_TOKEN=$2
HEAD_IP=$(hostname -i)

# Start head node in tmux session
tmux new-session -d -s ray_head " bash vllm/examples/online_serving/run_cluster.sh \
    $IMAGE_URI $HEAD_IP \
    /fsx/.cache/huggingface \
    -e VLLM_HOST_IP=$HEAD_IP \
    -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
    -e FI_PROVIDER=efa \
    -e FI_EFA_USE_DEVICE_RDMA=1 \
    --device=/dev/infiniband/ \
    --ulimit memlock=-1:-1 \
    -p 8000:8000"

sleep 30

CONTAINER_ID=$(docker ps --filter name=node-* --format "{{.ID}}" | head -n 1)

if [ -z "$CONTAINER_ID" ]; then
    echo "Failed to get container ID"
    exit 1
fi

echo "Head node container ID: $CONTAINER_ID"

tmux new-session -d -s vllm_serve "docker exec -it \$CONTAINER_ID /bin/bash -c \
'vllm serve ${model_name} \
--tensor-parallel-size 8 \
--pipeline-parallel-size 2 \
--max-num-batched-tokens 16384 \
--port 8000'"

echo "Head node and vllm serve started"
