<<<<<<< HEAD
#script for single node
=======
#!/bin/bash

set -e

INSTANCE_TYPE="p4d.24xlarge"
HF_TOKEN="example"
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
CONTAINER_IMAGE=""

# Function to clean up the container
cleanup() {
    echo "Cleaning up..."
    docker stop vllm-server 2>/dev/null || true
    docker rm vllm-server 2>/dev/null || true
}

# Set up trap to ensure cleanup on exit
trap cleanup EXIT

# Pull and run the vLLM DLC
echo "Starting vLLM server..."
docker run --name vllm-server --rm -d --runtime nvidia --gpus all \
    -v /fsx/.cache/huggingface:/root/.cache/huggingface \
    -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
    -e "NCCL_DEBUG=TRACE" \
    -p 8000:8000 \
    --ipc=host \
    ${CONTAINER_IMAGE} \
    --model ${MODEL_NAME} \
    --tensor-parallel-size 8

# Wait for the server to start
echo "Waiting for the server to start..."
until $(curl --output /dev/null --silent --fail http://localhost:8000/v1/models); do
    printf '.'
    sleep 10
done
echo -e "\nServer is up!"

# Additional delay to ensure model is fully loaded
echo "Waiting for model to fully load..."
sleep 30

# Perform inference checks
echo "Performing inference checks..."

# Completions API check
echo "Checking Completions API..."
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "prompt": "Hello, how are you?",
    "max_tokens": 100
  }'

echo -e "\n"

# Chat Completions API check
echo "Checking Chat Completions API..."
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "messages": [{"role": "user", "content": "Hello, how are you?"}]
  }'

echo -e "\n"
echo "Inference checks completed."

# Run the benchmark
echo "Starting benchmark..."
python3 /fsx/vllm-dlc/vllm/benchmarks/benchmark_serving.py \
  --backend vllm \
  --model ${MODEL_NAME} \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path /fsx/vllm-dlc/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 1000

echo "Benchmark completed."
>>>>>>> 83a249f9 (fix sg and fsx)
