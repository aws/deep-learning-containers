#!/bin/bash

DLC_IMAGE=$1
HF_TOKEN=$2
MODEL_NAME=$3

# Run vLLM using Official Docker image from https://docs.vllm.ai/en/latest/deployment/docker.html 
# Here is the https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile
tmux new-session -d -s single_node "docker run --runtime nvidia --gpus all \
    -v /fsx/.cache/huggingface:/root/.cache/huggingface \
    -e "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    -e "NCCL_DEBUG=TRACE" \
    -p 8000:8000 \
    --ipc=host \
    $DLC_IMAGE \
    --model $MODEL_NAME \
    --tensor-parallel-size 8"

sleep 1500

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "messages": [{"role": "user", "content": "Hello, how are you?"}]
  }'

python3 -m venv vllm_env
source vllm_env/bin/activate
pip install --upgrade pip setuptools wheel 
pip install numpy torch tqdm aiohttp pandas datasets pillow ray vllm==0.10.0
pip install "transformers<4.54.0"

# Example - Online Benchmark: https://github.com/vllm-project/vllm/tree/main/benchmarks#example---online-benchmark
python3 /fsx/vllm-dlc/vllm/benchmarks/benchmark_serving.py \
  --backend vllm \
  --model $MODEL_NAME \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path /fsx/vllm-dlc/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 1000

# ============ Serving Benchmark Result ============
# Successful requests:                     1000      
# Benchmark duration (s):                  82.67     
# Total input tokens:                      215196    
# Total generated tokens:                  185671    
# Request throughput (req/s):              12.10     
# Output token throughput (tok/s):         2245.92   
# Total Token throughput (tok/s):          4848.99   
# ---------------Time to First Token----------------
# Mean TTFT (ms):                          25037.89  
# Median TTFT (ms):                        22099.12  
# P99 TTFT (ms):                           58100.87  
# -----Time per Output Token (excl. 1st token)------
# Mean TPOT (ms):                          98.10     
# Median TPOT (ms):                        92.09     
# P99 TPOT (ms):                           256.34    
# ---------------Inter-token Latency----------------
# Mean ITL (ms):                           84.56     
# Median ITL (ms):                         63.78     
# P99 ITL (ms):                            253.97    
# ==================================================
