# #!/bin/bash

# set -e

# # Function to clean up the container
# cleanup() {
#     echo "Cleaning up..."
#     docker stop vllm-server 2>/dev/null || true
#     docker rm vllm-server 2>/dev/null || true
# }

# CONTAINER_IMAGE=$1
# HF_TOKEN=$2
# MODEL_NAME=$3

# # Set up trap to ensure cleanup on exit
# trap cleanup EXIT

# # Function to log messages with timestamps
# log() {
#     echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
# }

# log "Starting script execution"

# # Check if the model is already cached
# log "Checking if model is cached..."
# if [ -d "/fsx/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B" ]; then
#     log "Model is already cached"
# else
#     log "Model is not cached. It will be downloaded during server startup."
# fi

# python3 -m venv vllm_env 
# source vllm_env/bin/activate 
# pip install --upgrade pip setuptools wheel 
# pip install numpy torch tqdm aiohttp pandas datasets pillow vllm
# pip install "transformers[torch]" 
# echo "Python version: $(python --version)"

# # Set variables
# GPU_ID=0 # or 0,1 for multi-GPU
# PORT=8000
# MAX_MODEL_LEN=131072
# MAX_NUM_SEQS=256
# GPU_MEMORY_UTILIZATION=0.9 # default 0.9
# LOG_FILE="/home/ec2-user/vllm_docker.log"
# dtype=half

# # Calculate tensor parallel size based on the number of GPUs
# IFS=',' read -r -a GPU_ARRAY <<< "$GPU_ID"
# TENSOR_PARALLEL_SIZE=${#GPU_ARRAY[@]}

# log "Starting vllm openai container..."

# docker run --runtime nvidia --gpus all \
#     -v /fsx/.cache/huggingface:/root/.cache/huggingface \
#     -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
#     -e "NCCL_DEBUG=TRACE" \
#     -p 8000:8000 \
#     --ipc=host \
#     vllm/vllm-openai:latest \
#     --model ${MODEL_NAME} \
#     --tensor-parallel-size 8 

# log "Checking Completions API..."

# curl http://localhost:8000/v1/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
#     "prompt": "Hello, how are you?",
#     "max_tokens": 100
#   }'

# log "Starting vLLM DLC server..."

# # Run the Docker container
# docker run --name vllm-server --runtime nvidia --gpus all \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     -e "HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN" \
#     -e "NCCL_DEBUG=TRACE" \
#     -p $PORT:8000 \
#     --ipc=host \
#     ${CONTAINER_IMAGE} \
#     --model ${MODEL_NAME} \
#     --trust-remote-code \
#     --host 0.0.0.0 \
#     --max-model-len $MAX_MODEL_LEN \
#     --tensor-parallel-size 8 \

# # Pull and run the vLLM DLC
# # docker run --name vllm-server --runtime nvidia --gpus all \
# #     -v /fsx/.cache/huggingface:/root/.cache/huggingface \
# #     -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
# #     -e "NCCL_DEBUG=TRACE" \
# #     -p 8000:8000 \
# #     --ipc=host \
# #     ${CONTAINER_IMAGE} \
# #     --model ${MODEL_NAME} \
# #     --tensor-parallel-size 8 

# log "Checking container status..."
# if ! docker ps | grep -q vllm-server; then
#     log "Container failed to start. Logs:"
#     docker logs vllm-server
#     exit 1
# fi


# echo "Waiting for the server to start..."
# until $(curl --output /dev/null --silent --fail http://localhost:8000/v1/models); do
#   printf '.'
#   sleep 10
# done
# echo -e "\nServer is up!"

# # Additional delay to ensure model is fully loaded
# echo "Waiting for model to fully load..."
# sleep 30


# # Perform inference checks
# log "Performing inference checks..."

# # Completions API check
# log "Checking Completions API..."
# curl http://localhost:8000/v1/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
#     "prompt": "Hello, how are you?",
#     "max_tokens": 100
#   }'

# echo -e "\n"

# # Chat Completions API check
# log "Checking Chat Completions API..."
# curl http://localhost:8000/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
#     "messages": [{"role": "user", "content": "Hello, how are you?"}]
#   }'

# echo -e "\n"
# log "Inference checks completed."

# # Run the benchmark
# log "Starting benchmark..."
# python3 /fsx/vllm-dlc/vllm/benchmarks/benchmark_serving.py \
#   --backend vllm \
#   --base-url http://127.0.0.1:8080 \
#   --model ${MODEL_NAME} \
#   --endpoint /v1/completions \
#   --dataset-name sharegpt \
#   --dataset-path /fsx/vllm-dlc/ShareGPT_V3_unfiltered_cleaned_split.json \
#   --num-prompts 1000 \

# log "Benchmark completed."
# log "Script execution finished"


# #!/bin/bash

set -e

# Function to clean up the container
cleanup() {
    echo "Cleaning up..."
    docker stop vllm-server 2>/dev/null || true
    docker rm vllm-server 2>/dev/null || true
}

CONTAINER_IMAGE=$1
HF_TOKEN=$2
MODEL_NAME=$3

# Set up trap to ensure cleanup on exit
trap cleanup EXIT

INSTANCE_TYPE="p4d.24xlarge"
UPSTREAM_IMAGE="vllm/vllm-openai:latest"

# Function to run benchmark
run_benchmark() {
    local container_image=$1
    local output_file=$2
    local container_name="vllm-server-${RANDOM}"

    echo "Running benchmark for $container_image"
    docker run --name $container_name --rm -d --runtime nvidia --gpus all \
        -v /fsx/.cache/huggingface:/root/.cache/huggingface \
        -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
        -e "NCCL_DEBUG=TRACE" \
        -p 8000:8000 \
        --ipc=host \
        $container_image \
        --model ${MODEL_NAME} \
        --tensor-parallel-size 8

    echo "Waiting for the server to start..."
    until $(curl --output /dev/null --silent --fail http://localhost:8000/v1/models); do
        printf '.'
        sleep 10
    done
    echo -e "\nServer is up!"

    # Additional delay to ensure model is fully loaded
    echo "Waiting for model to fully load..."
    sleep 30

    echo "Running benchmark..."
    python3 /fsx/vllm-dlc/benchmarks/benchmark_serving.py \
      --backend vllm \
      --model ${MODEL_NAME} \
      --endpoint /v1/completions \
      --dataset-name sharegpt \
      --dataset-path /fsx/vllm-dlc/ShareGPT_V3_unfiltered_cleaned_split.json \
      --num-prompts 1000 > $output_file

    # Stop and remove the docker container
    docker stop $container_name
}

# Ensure cleanup on script exit
trap 'docker ps -q | xargs -r docker stop' EXIT

# Run benchmarks
run_benchmark "$UPSTREAM_IMAGE" "upstream_benchmark.txt"
run_benchmark "$CONTAINER_IMAGE" "dlc_benchmark.txt"

# Compare results
echo "vLLM DLC results:"
cat dlc_benchmark.txt
echo "--------------------"
echo "Upstream container results:"
cat upstream_benchmark.txt

