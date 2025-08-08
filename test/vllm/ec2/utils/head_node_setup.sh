#!/bin/bash

# Usage: ./head_node_setup.sh <image_uri> <hf_token>
set -e

IMAGE_URI=$1
HF_TOKEN=$2
HEAD_IP=$(hostname -i)

# Start head node in tmux session and capture container ID
tmux new-session -d -s ray_head "bash /fsx/vllm-dlc/vllm/examples/online_serving/run_cluster.sh \
    $IMAGE_URI $HEAD_IP \
    /fsx/.cache/huggingface \
    -e VLLM_HOST_IP=$HEAD_IP \
    -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
    -e FI_PROVIDER=efa \
    -e FI_EFA_USE_DEVICE_RDMA=1 \
    --device=/dev/infiniband/ \
    --ulimit memlock=-1:-1 \
    -p 8000:8000"

echo "Waiting for container to start..."
sleep 200

docker ps -a

CONTAINER_ID=$(docker ps --format '{{.ID}} {{.Names}}' | awk '/node-/ {print $1}' | head -n 1)
echo "Head node container ID: $CONTAINER_ID"

tmux new-session -d -s vllm_serve "docker exec -it $CONTAINER_ID /bin/bash -c \
'vllm serve ${model_name} \
--tensor-parallel-size 8 \
--pipeline-parallel-size 2 \
--max-num-batched-tokens 16384 \
--port 8000'"

echo "Head node and vllm serve started"
