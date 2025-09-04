#!/bin/bash
set -e

DLC_IMAGE=$1
HF_TOKEN=$2

if [ -z "$DLC_IMAGE" ] || [ -z "$HF_TOKEN" ]; then
    echo "Usage: $0 <dlc-image> <hugging-face-token>"
    exit 1
fi

echo "🚀 Starting VLLM testing pipeline..."

wait_for_api() {
    echo "Waiting for VLLM API to be ready..."
    while ! curl -s "http://localhost:8000/v1/health" > /dev/null; do
        sleep 5
        echo "Still waiting for API..."
    done
    echo "API is ready!"
}

# Cleanup function
cleanup() {
    echo "Cleaning up containers..."
    docker stop vllm-arm64-dlc || true
    docker rm vllm-arm64-dlc || true
}

trap cleanup EXIT

echo "📝 Running initial inference check..."

# Initial inference test
docker run --rm -v /fsx/vllm-dlc/vllm:/vllm \
    -e "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --gpus=all \
    --entrypoint="" \
    $DLC_IMAGE \
    bash -c 'python3 /vllm/examples/offline_inference/basic/generate.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --dtype half \
    --tensor-parallel-size 1 \
    --max-model-len 2048'

echo "🤖 Starting VLLM server..."

docker run -d \
    --name vllm-arm64-dlc \
    --runtime nvidia \
    --gpus all \
    -e "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    -e "VLLM_WORKER_MULTIPROC_METHOD=spawn" \
    -e "NCCL_DEBUG=TRACE" \
    -p 8000:8000 \
    --ipc=host \
    $DLC_IMAGE \
    bash -c 'python3 -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --tensor-parallel-size 2 \
    --dtype half'

wait_for_api

echo "Testing API endpoint..."

curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "prompt": "Hello",
        "max_tokens": 10
    }'

echo "Installing Python dependencies..."

python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install openai 'strands-agents[openai]' strands-agents-tools

echo "Running agent tests..."

# Run agent tests
python3 test_agents.py

echo "Testing completed successfully!"

deactivate

