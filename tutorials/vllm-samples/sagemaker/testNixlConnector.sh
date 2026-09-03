#!/bin/bash

# Function to wait for server to be ready
wait_for_server() {
    local host=$1
    local port=$2
    local timeout=120
    local count=0
    
    echo "Waiting for server at $host:$port to be ready..."
    while ! curl -s http://$host:$port/health > /dev/null 2>&1; do
        sleep 5
        count=$((count + 5))
        if [ $count -ge $timeout ]; then
            echo "Timeout waiting for server at $host:$port"
            return 1
        fi
    done
    echo "Server at $host:$port is ready"
}

# Start first GPU (prefiller)
echo "Starting prefiller on GPU 0..."
CUDA_VISIBLE_DEVICES=0 \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --port 8100 \
  --max-model-len 6000 \
  --enforce-eager \        
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'

# Start second GPU (decoder)
echo "Starting decoder on GPU 1..."
CUDA_VISIBLE_DEVICES=1 \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --port 8200 \
  --max-model-len 6000 \
  --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'


# Wait for GPU servers
wait_for_server localhost 8100
wait_for_server localhost 8200

# Start proxy server
echo "Starting proxy server..."
python3 proxy.py \
  --host 0.0.0.0 \
  --port 8192 \
  --prefiller-hosts localhost \
  --prefiller-ports 8100 \
  --decoder-hosts localhost \
  --decoder-ports 8200

# Wait for proxy server
wait_for_server localhost 8192

# wget -q https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
vllm bench serve \
    --host 0.0.0.0 \
    --port 8192 \
    --model  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 30
    
