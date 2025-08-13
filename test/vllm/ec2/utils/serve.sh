#!/bin/bash

CONTAINER_NAME=$1
MODEL_NAME=$2
start_time=$(date +%s)
timeout=1800
LOG_FILE="vllm_$(date +%Y%m%d_%H%M%S).log"

echo "Starting vllm service... (this will take ~20 minutes)" | tee $LOG_FILE
tmux new-session -d -s vllm_serve "docker exec -it $CONTAINER_NAME /bin/bash -c 'vllm serve $MODEL_NAME \
--tensor-parallel-size 8 \
--pipeline-parallel-size 2 \
--max-num-batched-tokens 16384' 2>&1 | tee -a $LOG_FILE"

while true; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    
    if [ $elapsed -ge $timeout ]; then
        echo "Timeout reached (30 minutes). Breaking loop."
        break
    fi
    
    echo "Still loading..." | tee -a $LOG_FILE
    sleep 60
done
echo "Model is ready" | tee -a $LOG_FILE

source vllm_env/bin/activate 
echo "Starting benchmark..." | tee -a $LOG_FILE

python3 /fsx/vllm-dlc/vllm/benchmarks/benchmark_serving.py \
    --backend vllm \
    --model $MODEL_NAME \
    --endpoint /v1/chat/completions \
    --dataset-name sharegpt \
    --dataset-path /fsx/vllm-dlc/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 2>&1 | tee -a $LOG_FILE

echo "Benchmark complete" | tee -a $LOG_FILE
