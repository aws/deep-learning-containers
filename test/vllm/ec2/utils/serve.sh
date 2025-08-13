#!/bin/bash

CONTAINER_NAME=$1
MODEL_NAME=$2

echo "Starting vllm service... (this will take ~20 minutes)"
tmux new-session -d -s vllm_serve "docker exec -it $CONTAINER_NAME /bin/bash -c 'vllm serve $MODEL_NAME \
--tensor-parallel-size 8 \
--pipeline-parallel-size 2 \
--max-num-batched-tokens 16384'"

while ! curl -s "http://localhost:8000/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "prompt": "Hello", "max_tokens": 10}' \
    > /dev/null 2>&1; do
    echo "Still loading..."
    sleep 60
done

echo "Model is ready"
