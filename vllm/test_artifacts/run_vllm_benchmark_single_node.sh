#!/bin/bash

set -e

# Function to clean up the container
cleanup() {
    echo "Cleaning up..."
    docker stop vllm-server 2>/dev/null || true
    docker rm vllm-server 2>/dev/null || true
}

# Set up trap to ensure cleanup on exit
trap cleanup EXIT

# Function to log messages with timestamps
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "Starting script execution"

# Check if the model is already cached
log "Checking if model is cached..."
if [ -d "/fsx/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B" ]; then
    log "Model is already cached"
else
    log "Model is not cached. It will be downloaded during server startup."
fi

# Pull and run the vLLM DLC
log "Starting vLLM server..."
docker run --name vllm-server --rm -d --runtime nvidia --gpus all \
    -v /fsx/.cache/huggingface:/root/.cache/huggingface \
    -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
    -e "NCCL_DEBUG=TRACE" \
    -e "TRANSFORMERS_VERBOSITY=info" \
    -e "HF_HUB_ENABLE_HF_TRANSFER=1" \
    -p 8000:8000 \
    --ipc=host \
    ${CONTAINER_IMAGE} \
    --model ${MODEL_NAME} \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --max-log-level debug

# Function to check server status
check_server() {
    if curl --output /dev/null --silent --fail http://localhost:8000/v1/models; then
        return 0
    else
        return 1
    fi
}

# Wait for the server to start
log "Waiting for the server to start..."
start_time=$(date +%s)
while ! check_server; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $elapsed -ge 1800 ]; then  # 30 minutes timeout
        log "Server failed to start within 30 minutes. Exiting."
        docker logs vllm-server
        exit 1
    fi
    log "Server not ready. Elapsed time: ${elapsed}s. Waiting..."
    sleep 30
done
log "Server is up! Total startup time: $(($(date +%s) - start_time)) seconds"

# Additional delay to ensure model is fully loaded
log "Waiting for model to fully load..."
sleep 30

# Perform inference checks
log "Performing inference checks..."

# Completions API check
log "Checking Completions API..."
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "prompt": "Hello, how are you?",
    "max_tokens": 100
  }'

echo -e "\n"

# Chat Completions API check
log "Checking Chat Completions API..."
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "messages": [{"role": "user", "content": "Hello, how are you?"}]
  }'

echo -e "\n"
log "Inference checks completed."

# Run the benchmark
log "Starting benchmark..."
python3 /fsx/vllm/vllm/benchmarks/benchmark_serving.py \
  --backend vllm \
  --model ${MODEL_NAME} \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path /fsx/vllm/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 1000

log "Benchmark completed."
log "Script execution finished"
