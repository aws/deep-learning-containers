#!/bin/bash

# Usage: ./head_node_setup.sh <image_uri> <hf_token> <model_name>
set -e

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

if [ "$#" -ne 3 ]; then
    log "Error: Required parameters missing"
    log "Usage: $0 <image_uri> <hf_token> <model_name>"
    exit 1
fi

IMAGE_URI=$1
HF_TOKEN=$2
MODEL_NAME=$3
HEAD_IP=$(hostname -i)
WORKER_IP=$(ssh compute2 "hostname -I" | awk '{print $1}')

log "Starting cluster setup..."
log "Image URI: $IMAGE_URI"
log "Head IP: $HEAD_IP"
log "Worker IP: $WORKER_IP"

# Start head node in tmux session and capture container ID
log "Starting head node..."
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

# Wait for head node to start and get container ID
sleep 10
HEAD_CONTAINER_ID=$(docker ps -q --filter "ancestor=$IMAGE_URI" --filter "status=running" | head -n 1)

if [ -z "$HEAD_CONTAINER_ID" ]; then
    log "Error: Failed to get head container ID"
    exit 1
fi

log "Head node started with container ID: $HEAD_CONTAINER_ID"

# Start worker node via SSH
log "Starting worker node..."
ssh compute2 "tmux new-session -d -s ray_worker 'bash /fsx/vllm-dlc/vllm/examples/online_serving/run_cluster.sh \
    $IMAGE_URI \
    $HEAD_IP \
    --worker \
    /fsx/.cache/huggingface \
    -e VLLM_HOST_IP=$WORKER_IP \
    -e FI_PROVIDER=efa \
    -e FI_EFA_USE_DEVICE_RDMA=1 \
    --device=/dev/infiniband/ \
    --ulimit memlock=-1:-1'"

log "Worker node setup initiated"

# Wait for worker to connect
sleep 20

# Start vllm serve on head node
log "Starting vLLM serve..."
docker exec -it $HEAD_CONTAINER_ID /bin/bash -c "vllm serve $MODEL_NAME \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --max-num-batched-tokens 16384"

sleep 1000

log "vLLM serve started"
log "vLLM service should now be running on port 8000"

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "messages": [{"role": "user", "content": "Hello, how are you?"}]
  }'
  
log "Setup complete. vLLM service should now be running."
