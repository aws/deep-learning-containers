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

# NixlConnector ignores kv_role at the engine level: the per-request
# kv_transfer_params dict from the proxy is what determines remote-fetch
# behavior. kv_both matches upstream's run_accuracy_test.sh.
#
# kv_load_failure_policy=fail makes a missing/incomplete handoff a hard error
# instead of a silent local re-prefill — so a transport regression surfaces
# immediately rather than passing as a coherent completion.
KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_connector_extra_config":{"backends":["LIBFABRIC"],"kv_load_failure_policy":"fail"}}'

# Side-channel host: needs to be the IP that the worker can reach this box on.
SIDE_CHANNEL_HOST=$(ip -4 -o addr show scope global | awk '{print $4}' | cut -d/ -f1 | head -1)

# Block size must match across P and D for remote_block_ids to map correctly;
# upstream's NIXL accuracy test pins this to 128 (OPT's default is 16, which
# breaks the lookup at scheduler.py if D defaults differently).
BLOCK_SIZE=128

cleanup() {
    set +e
    [ -n "${PREFILL_PID}" ] && kill -TERM "${PREFILL_PID}" 2>/dev/null
    [ -n "${PROXY_PID}" ] && kill -TERM "${PROXY_PID}" 2>/dev/null
    wait 2>/dev/null
}
trap cleanup EXIT

# --- Launch prefill server on master ---
# VLLM_KV_CACHE_LAYOUT=HND is required by NixlConnector and matches upstream.
CUDA_VISIBLE_DEVICES=0 \
FI_PROVIDER=efa \
VLLM_KV_CACHE_LAYOUT=HND \
VLLM_NIXL_SIDE_CHANNEL_PORT=${SIDE_CHANNEL_PORT} \
VLLM_NIXL_SIDE_CHANNEL_HOST=${SIDE_CHANNEL_HOST} \
vllm serve "${MODEL}" \
    --port ${PREFILL_PORT} \
    --enforce-eager \
    --block-size ${BLOCK_SIZE} \
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

# --- Strict KV-transfer assertion via NixlConnector Prometheus metrics ---
# vllm:nixl_xfer_time_seconds_count is a histogram counter that increments per
# successful NIXL transfer. > 0 means at least one KV cache block crossed the
# wire from prefill to decode over libfabric/EFA. vllm:nixl_num_failed_transfers
# is a Counter that must stay at 0.
#
# vllm:prompt_tokens_total is NOT a useful proof: it always counts the prompt
# size regardless of whether the KV came from cache or local recompute.
_metric_value() {
    # Sum the value of a Prometheus metric, summing across labels if any.
    local url="$1" name="$2"
    curl -sf "${url}" | awk -v n="${name}" '
        $0 ~ "^"n"[ {]" { gsub(/.*[ ]/, "", $0); s += $0 + 0 }
        END { printf "%d", s }
    '
}

DECODE_XFERS=$(_metric_value "http://${WORKER_IP}:${DECODE_PORT}/metrics" "vllm:nixl_xfer_time_seconds_count")
DECODE_FAILED=$(_metric_value "http://${WORKER_IP}:${DECODE_PORT}/metrics" "vllm:nixl_num_failed_transfers")
DECODE_GEN=$(_metric_value "http://${WORKER_IP}:${DECODE_PORT}/metrics" "vllm:generation_tokens_total")
echo "metrics: decode.nixl_xfers=${DECODE_XFERS} decode.nixl_failed=${DECODE_FAILED} decode.gen_tokens=${DECODE_GEN}"

[ "${DECODE_XFERS}" -ge 1 ] || { echo "no NIXL transfers reached decode (xfer_count=${DECODE_XFERS}) — KV did not flow over libfabric"; exit 1; }
[ "${DECODE_FAILED}" -eq 0 ] || { echo "NIXL transfers failed (failed=${DECODE_FAILED})"; exit 1; }
[ "${DECODE_GEN}" -ge 1 ] || { echo "decode produced no tokens (gen_tokens=${DECODE_GEN})"; exit 1; }

echo "nixl_disagg_pd test passed (${DECODE_XFERS} NIXL transfer(s) verified via /metrics)"
