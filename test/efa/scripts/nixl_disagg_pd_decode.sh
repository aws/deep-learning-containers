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

# kv_consumer: decode side refuses to prefill locally. If KV bytes don't
# arrive from prefill over libfabric, the request hangs and the orchestrator's
# curl times out — turning silent fallback into a hard test failure.
KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"backends":["LIBFABRIC"]}}'

# Daemonize so the SSH/docker-exec call that launched us returns immediately;
# the orchestrator polls /health to know when we're actually ready.
nohup env CUDA_VISIBLE_DEVICES=0 \
    FI_PROVIDER=efa \
    UCX_NET_DEVICES=all \
    VLLM_NIXL_SIDE_CHANNEL_PORT=${SIDE_CHANNEL_PORT} \
    vllm serve "${MODEL}" \
        --host 0.0.0.0 \
        --port ${DECODE_PORT} \
        --enforce-eager \
        --gpu-memory-utilization 0.5 \
        --kv-transfer-config "${KV_CONFIG}" \
    >"${DECODE_LOG}" 2>&1 &
echo $! >"${LOG_DIR}/decode.pid"
disown
sleep 1
