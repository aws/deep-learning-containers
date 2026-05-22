#!/bin/bash
# Decode-side launcher for the NIXL disaggregated PD EFA test.
# Runs on the worker node; the master orchestrator starts it via SSH and waits
# for /health on port 8200. Uses NIXL with the LIBFABRIC backend so KV cache
# transfer from the prefill node arrives over EFA.
#
# Args:
#   $1 = MODEL — HF model id (default: facebook/opt-125m)
set -ex

MODEL=${1:-facebook/opt-125m}

LOG_DIR=/test/efa/logs
mkdir -p "${LOG_DIR}"
DECODE_LOG="${LOG_DIR}/decode.log"

DECODE_PORT=8200
SIDE_CHANNEL_PORT=5659

# NixlConnector ignores kv_role at the engine level: the per-request
# kv_transfer_params dict from the proxy is what determines remote-fetch
# behavior. kv_both matches upstream's run_accuracy_test.sh.
#
# kv_load_failure_policy=fail surfaces incomplete KV handoffs as hard errors
# instead of silent local re-prefill.
KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_connector_extra_config":{"backends":["LIBFABRIC"],"kv_load_failure_policy":"fail"}}'

# Side-channel host must be reachable from the prefill node (cross-host pull).
# Default to the first non-loopback IPv4 address on this box.
SIDE_CHANNEL_HOST=$(ip -4 -o addr show scope global | awk '{print $4}' | cut -d/ -f1 | head -1)

# Block size must match prefill (see nixl_disagg_pd.sh).
BLOCK_SIZE=128

# Daemonize so the SSH/docker-exec call that launched us returns immediately;
# the orchestrator polls /health to know when we're actually ready.
nohup env CUDA_VISIBLE_DEVICES=0 \
    FI_PROVIDER=efa \
    VLLM_KV_CACHE_LAYOUT=HND \
    VLLM_NIXL_SIDE_CHANNEL_PORT=${SIDE_CHANNEL_PORT} \
    VLLM_NIXL_SIDE_CHANNEL_HOST=${SIDE_CHANNEL_HOST} \
    vllm serve "${MODEL}" \
        --host 0.0.0.0 \
        --port ${DECODE_PORT} \
        --enforce-eager \
        --block-size ${BLOCK_SIZE} \
        --gpu-memory-utilization 0.5 \
        --kv-transfer-config "${KV_CONFIG}" \
    >"${DECODE_LOG}" 2>&1 &
echo $! >"${LOG_DIR}/decode.pid"
disown
sleep 1
