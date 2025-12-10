#!/bin/bash

#==============================================================================
# VLLM NIXL Integration Test Script
#==============================================================================
# This script tests the NIXL KV connector with LIBFABRIC backend by:
# 1. Starting a prefiller service on GPU 0
# 2. Starting a decoder service on GPU 1
# 3. Running a proxy server to coordinate between them
# 4. Executing a benchmark with ShareGPT dataset
#==============================================================================

#------------------------------------------------------------------------------
# Configuration Variables
#------------------------------------------------------------------------------
IMAGE=$1
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both", "kv_buffer_device":"cuda","kv_connector_extra_config":{"backends":["LIBFABRIC"]}}'
LOG_DIR="/fsx/vllm-dlc/logs"

#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------
mkdir -p "$LOG_DIR"
cd /fsx/vllm-dlc/ || exit 1
pip install vllm

#------------------------------------------------------------------------------
# Helper Functions
#------------------------------------------------------------------------------

# Check if LIBFABRIC backend was successfully instantiated in service logs
check_libfabric() {
    local log_file=$1
    local service=$2
    
    echo "Checking $service logs for LIBFABRIC backend..."
    
    if grep -q "Backend LIBFABRIC was instantiated" "$log_file"; then
        echo "✅ $service: LIBFABRIC backend successfully instantiated"
    else
        echo "❌ $service: LIBFABRIC backend not found in logs"
        echo "Last 50 lines of $service log:"
        tail -n 50 "$log_file"
    fi
}

# Wait for a service to become healthy by polling its health endpoint
wait_for_server() {
    local port=$1
    local service=$2
    local log_file=$3
    local max_attempts=60
    local attempt=1

    echo "Waiting for $service to be ready on port $port..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "http://localhost:$port/health" >/dev/null; then
            echo "✅ $service is ready (attempt $attempt)"
            return 0
        fi
        sleep 5
        attempt=$((attempt + 1))
    done
    
    echo "❌ Timeout waiting for $service after $max_attempts attempts"
    cat "$log_file"
    return 1
}

#------------------------------------------------------------------------------
# Start Prefiller Service (GPU 0)
#------------------------------------------------------------------------------
echo "Starting prefiller on GPU 0..."

docker run --rm \
    --gpus all \
    --network host \
    --device=/dev/infiniband \
    --ulimit memlock=-1 \
    --entrypoint=/bin/bash \
    --name vllm-prefill \
    -v /fsx/.cache/huggingface:/root/.cache/huggingface \
    "$IMAGE" \
    -c "CUDA_VISIBLE_DEVICES=0,1 \
        UCX_NET_DEVICES=all \
        VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
        vllm serve $MODEL \
        --port 8100 \
        --max-model-len 6000 \
        --enforce-eager \
        --kv-transfer-config '$KV_CONFIG'" \
    > "$LOG_DIR/prefill.log" 2>&1 &

wait_for_server 8100 "Prefiller" "$LOG_DIR/prefill.log" || exit 1
check_libfabric "$LOG_DIR/prefill.log" "Prefiller"

#------------------------------------------------------------------------------
# Start Decoder Service (GPU 1)
#------------------------------------------------------------------------------
echo "Starting decoder on GPU 1..."

docker run --rm \
    --gpus all \
    --network host \
    --device=/dev/infiniband \
    --ulimit memlock=-1 \
    --entrypoint=/bin/bash \
    --name vllm-decoder \
    -v /fsx/.cache/huggingface:/root/.cache/huggingface \
    "$IMAGE" \
    -c "CUDA_VISIBLE_DEVICES=1,2 \
        UCX_NET_DEVICES=all \
        VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
        vllm serve $MODEL \
        --port 8200 \
        --enforce-eager \
        --kv-transfer-config '$KV_CONFIG'" \
    > "$LOG_DIR/decode.log" 2>&1 &

wait_for_server 8200 "Decoder" "$LOG_DIR/decode.log" || exit 1
check_libfabric "$LOG_DIR/decode.log" "Decoder"

#------------------------------------------------------------------------------
# Start Proxy Server
#------------------------------------------------------------------------------
echo "Starting proxy server..."

python3 vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
    --host 0.0.0.0 \
    --port 8192 \
    --prefiller-hosts localhost \
    --prefiller-ports 8100 \
    --decoder-hosts localhost \
    --decoder-ports 8200 \
    > "$LOG_DIR/proxy.log" 2>&1 &

#------------------------------------------------------------------------------
# Run Benchmark
#------------------------------------------------------------------------------
echo "Starting benchmark..."

vllm bench serve \
    --host 0.0.0.0 \
    --port 8192 \
    --model "$MODEL" \
    --dataset-name sharegpt \
    --dataset-path /fsx/vllm-dlc/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 30 \
    | tee "$LOG_DIR/benchmark.log"

#------------------------------------------------------------------------------
# Print Service Logs
#------------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "PREFILLER LOGS:"
echo "=========================================="
cat "$LOG_DIR/prefill.log"

echo ""
echo "=========================================="
echo "DECODER LOGS:"
echo "=========================================="
cat "$LOG_DIR/decode.log"

#------------------------------------------------------------------------------
# Cleanup
#------------------------------------------------------------------------------
cleanup() {
    echo "Cleaning up containers..."
    docker stop vllm-prefill vllm-decoder >/dev/null 2>&1
}

trap cleanup EXIT
