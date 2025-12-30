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

cd /fsx/vllm-dlc/vllm
git checkout v0.10.2

wait_for_api() {
    local max_attempts=60
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
            docker logs ${CONTAINER_NAME}
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

echo "####################### RUNNING INFERENCE CHECK ########################################"

docker run --rm \
    -v /fsx/vllm-dlc/vllm:/vllm \
    --entrypoint /bin/bash \
    -e "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    -e "VLLM_WORKER_MULTIPROC_METHOD=spawn" \
    -e "VLLM_USE_V1=0" \
    -v /fsx/.cache/huggingface:/root/.cache/huggingface \
    --gpus=all \
    $DLC_IMAGE \
    -c "python3 /vllm/examples/offline_inference/basic/generate.py \
        --model ${MODEL_NAME} \
        --dtype float16 \
        --tensor-parallel-size 1 \
        --max-model-len 2048"

echo "####################### Starting VLLM server ##########################################"

docker run -d \
    -v /fsx/vllm-dlc/vllm:/vllm \
    --name ${CONTAINER_NAME} \
    -p ${PORT}:8000 \
    --entrypoint /bin/bash \
    -e "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    -e "VLLM_WORKER_MULTIPROC_METHOD=spawn" \
    -e "VLLM_USE_V1=0" \
    -v /fsx/.cache/huggingface:/root/.cache/huggingface \
    --gpus=all \
    $DLC_IMAGE \
    -c "vllm serve ${MODEL_NAME} \
        --dtype float16 \
        --gpu-memory-utilization 0.7 \
        --max-model-len 6000 \
        --enforce-eager \
        --reasoning-parser deepseek_r1"

wait_for_api
docker logs "${CONTAINER_NAME}"

echo "####################### API TESTING ###########################"

curl -s http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "prompt": "What is AWS Deep learning container?",
            "max_tokens": 50
            }'

echo "####################### TESTING TOOL CALLS (OPEN AI API) ###########################"

python -m venv .venv
source .venv/bin/activate  

pip install "openai>=1.0.0"
python3 /fsx/vllm-dlc/vllm/examples/online_serving/openai_chat_completion_with_reasoning.py
deactivate

echo "####################### Testing completed successfully ###########################"
