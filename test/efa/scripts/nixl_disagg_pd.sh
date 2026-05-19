#!/bin/bash
# Two-node disaggregated prefill/decode test using NIXL with the LIBFABRIC
# backend over EFA. Boots a prefill vLLM server on this (master) host, expects
# the worker host to already have a decode server running (the orchestrator
# starts both via SSH), launches a small proxy locally, sends a completion
# request, and verifies the response is non-empty + libfabric/EFA appeared in
# the prefill log.
#
# Args:
#   $1 = WORKER_IP — private IP of the decode node
#   $2 = MODEL     — HF model id (default: facebook/opt-125m)
#
# Env vars used by the prefill side:
#   FI_PROVIDER=efa       — NIXL libfabric plugin selects EFA provider
#   UCX_NET_DEVICES=all   — match upstream test invocation
set -ex

WORKER_IP=$1
MODEL=${2:-facebook/opt-125m}

if [[ -z "${WORKER_IP}" ]]; then
    echo "usage: $0 <worker_ip> [model]" >&2
    exit 2
fi

LOG_DIR=/test/efa/logs
mkdir -p "${LOG_DIR}"
PREFILL_LOG="${LOG_DIR}/prefill.log"
PROXY_LOG="${LOG_DIR}/proxy.log"

PREFILL_PORT=8100
DECODE_PORT=8200
PROXY_PORT=8192
SIDE_CHANNEL_PORT=5559

KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_connector_extra_config":{"backends":["LIBFABRIC"]}}'

cleanup() {
    set +e
    [ -n "${PREFILL_PID}" ] && kill -TERM "${PREFILL_PID}" 2>/dev/null
    [ -n "${PROXY_PID}" ] && kill -TERM "${PROXY_PID}" 2>/dev/null
    wait 2>/dev/null
}
trap cleanup EXIT

# --- Launch prefill server on master ---
CUDA_VISIBLE_DEVICES=0 \
FI_PROVIDER=efa \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=${SIDE_CHANNEL_PORT} \
vllm serve "${MODEL}" \
    --port ${PREFILL_PORT} \
    --enforce-eager \
    --gpu-memory-utilization 0.5 \
    --kv-transfer-config "${KV_CONFIG}" \
    >"${PREFILL_LOG}" 2>&1 &
PREFILL_PID=$!

# --- Wait for prefill to be ready (max ~3 min) ---
for i in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:${PREFILL_PORT}/health" >/dev/null; then
        echo "prefill ready"
        break
    fi
    if ! kill -0 "${PREFILL_PID}" 2>/dev/null; then
        echo "prefill exited unexpectedly"
        tail -50 "${PREFILL_LOG}"
        exit 1
    fi
    sleep 2
done
curl -sf "http://127.0.0.1:${PREFILL_PORT}/health" >/dev/null || \
    { echo "prefill never came up"; tail -100 "${PREFILL_LOG}"; exit 1; }

# Wait for decode server on worker to be ready (started independently via SSH).
for i in $(seq 1 90); do
    if curl -sf "http://${WORKER_IP}:${DECODE_PORT}/health" >/dev/null; then
        echo "decode ready"
        break
    fi
    sleep 2
done
curl -sf "http://${WORKER_IP}:${DECODE_PORT}/health" >/dev/null || \
    { echo "decode never came up"; exit 1; }

# --- Launch proxy ---
python3 /test/efa/scripts/toy_proxy_server.py \
    --port ${PROXY_PORT} \
    --prefill-url "http://127.0.0.1:${PREFILL_PORT}" \
    --decode-url "http://${WORKER_IP}:${DECODE_PORT}" \
    >"${PROXY_LOG}" 2>&1 &
PROXY_PID=$!
sleep 3

# --- Send a completion request ---
RESPONSE=$(curl -sf -X POST "http://127.0.0.1:${PROXY_PORT}/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"The capital of France is\",\"max_tokens\":8,\"temperature\":0}")

echo "response: ${RESPONSE}"
echo "${RESPONSE}" | grep -q '"text"' || { echo "no completion text in response"; exit 1; }

# --- Validate NIXL+EFA actually engaged ---
grep -E "LIBFABRIC|libfabric" "${PREFILL_LOG}" || \
    { echo "no libfabric mention in prefill log"; tail -200 "${PREFILL_LOG}"; exit 1; }
grep -E "FI_EP_RDM|provider.*efa|Selected provider is efa" "${PREFILL_LOG}" || \
    echo "WARNING: couldn't confirm EFA provider was selected (may be in worker log)"

echo "nixl_disagg_pd test passed"
