#!/bin/bash
set -e

DLC_IMAGE=$1
HF_TOKEN=$2

if [ -z "$DLC_IMAGE" ] || [ -z "$HF_TOKEN" ]; then
    echo "Usage: $0 <dlc-image> <hugging-face-token>"
    exit 1
fi

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
CONTAINER_NAME="vllm-arm64-dlc"
PORT=8000

wait_for_api() {
    local max_attempts=30
    local attempt=1
    
    echo "Waiting for VLLM API to be ready..."
    while ! curl -s http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "prompt": "What is vllm?",
            "max_tokens": 30
            }' > /dev/null; do
        if [ $attempt -ge $max_attempts ]; then
            echo "Error: API failed to start after $max_attempts attempts"
            exit 1
        fi
        sleep 5
        ((attempt++))
    done
    echo "API is ready!"
}

cleanup() {
    echo "Cleaning up containers..."
    docker stop ${CONTAINER_NAME} 2>/dev/null || true
    docker rm ${CONTAINER_NAME} 2>/dev/null || true
}

handle_error() {
    echo "Error occurred on line $1"
    cleanup
    exit 1
}

trap cleanup EXIT
trap 'handle_error $LINENO' ERR

cd /fsx/vllm-dlc/vllm
git checkout main

echo "Running initial inference check..."
docker run --rm \
    -v /fsx/vllm-dlc/vllm:/vllm \
    --entrypoint /bin/bash \
    -e "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_FLASH_ATTN_VERSION=2 \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    --gpus=all \
    "$DLC_IMAGE" \
    -c "python3 /vllm/examples/offline_inference/basic/generate.py \
        --model ${MODEL_NAME} \
        --dtype half \
        --tensor-parallel-size 1 \
        --max-model-len 2048"

echo "Starting VLLM server..."
docker run -d \
    --entrypoint /bin/bash \
    --name ${CONTAINER_NAME} \
    --runtime nvidia \
    --gpus all \
    -e "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    -e "VLLM_WORKER_MULTIPROC_METHOD=spawn" \
    -e "NCCL_DEBUG=TRACE" \
    -p ${PORT}:${PORT} \
    --ipc=host \
    "$DLC_IMAGE" \
    -c "python3 -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --tensor-parallel-size 2 \
    --dtype float16"

wait_for_api
docker logs "${CONTAINER_NAME}"

# echo "VLLM server is running and responding to requests!"

# echo "Installing Python dependencies..."
# python -m venv .venv
# source .venv/bin/activate  

# pip install openai
# pip install strands-agents strands-agents-tools

# echo "Running agent tests..."
# python3 test_agents.py
# echo "Testing completed successfully!"

# deactivate