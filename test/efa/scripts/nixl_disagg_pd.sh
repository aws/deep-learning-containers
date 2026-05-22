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

# NixlConnector ignores kv_role at the engine level (v0.21.0): the per-request
# kv_transfer_params dict (do_remote_prefill / remote_engine_id / remote_block_ids
# / remote_host / remote_port) is what makes the decoder fetch instead of
# recompute. The proxy is responsible for shipping that handshake; the role
# is advisory only, so kv_both matches upstream's run_accuracy_test.sh.
KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_connector_extra_config":{"backends":["LIBFABRIC"]}}'

# Side-channel host: needs to be the IP that the worker can reach this box on.
SIDE_CHANNEL_HOST=$(ip -4 -o addr show scope global | awk '{print $4}' | cut -d/ -f1 | head -1)

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
VLLM_NIXL_SIDE_CHANNEL_HOST=${SIDE_CHANNEL_HOST} \
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

# --- Strict KV-transfer assertion via Prometheus metrics ---
# When the proxy ships kv_transfer_params correctly, decode reuses the prefilled
# KV cache and only processes the last token before generation (its scheduler's
# get_num_new_matched_tokens reports the prompt as already-cached). Asserting
# decode.prompt_tokens stays well below the full 6-token prompt proves KV bytes
# crossed the libfabric wire — without the connector, decode would re-prefill
# all 6 tokens locally.
_metric_value() {
    # Sum the value of a Prometheus counter, stripping comments + labels.
    local url="$1" name="$2"
    curl -sf "${url}" | awk -v n="${name}" '
        $0 ~ "^"n"[ {]" { gsub(/.*[ ]/, "", $0); s += $0 + 0 }
        END { printf "%d", s }
    '
}

PREFILL_PROMPT=$(_metric_value "http://127.0.0.1:${PREFILL_PORT}/metrics" "vllm:prompt_tokens_total")
DECODE_PROMPT=$(_metric_value "http://${WORKER_IP}:${DECODE_PORT}/metrics" "vllm:prompt_tokens_total")
DECODE_GEN=$(_metric_value "http://${WORKER_IP}:${DECODE_PORT}/metrics" "vllm:generation_tokens_total")
echo "metrics: prefill.prompt_tokens=${PREFILL_PROMPT} decode.prompt_tokens=${DECODE_PROMPT} decode.gen_tokens=${DECODE_GEN}"

[ "${PREFILL_PROMPT}" -ge 6 ] || { echo "prefill did not process prompt tokens (got ${PREFILL_PROMPT}, expected >=6)"; exit 1; }
# Decode should re-process at most 1 token (the last) when KV transfer worked.
# A value of 6 (the whole prompt) means the connector handshake failed.
[ "${DECODE_PROMPT}" -le 1 ] || { echo "decode re-prefilled (prompt_tokens=${DECODE_PROMPT}, expected <=1) — KV transfer over libfabric did not happen"; exit 1; }
[ "${DECODE_GEN}" -ge 1 ] || { echo "decode produced no tokens (gen_tokens=${DECODE_GEN})"; exit 1; }

echo "nixl_disagg_pd test passed (KV transfer verified via /metrics)"
