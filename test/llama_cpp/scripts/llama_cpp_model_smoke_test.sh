#!/bin/bash
set -euo pipefail

# llama.cpp Model Smoke Test
# Serves a local GGUF with llama-server and validates /health + a chat completion.
# Usage: llama_cpp_model_smoke_test.sh <model_path> <model_name> [extra_args...]

MODEL_PATH="${1:?Usage: $0 <model_path> <model_name> [extra_args...]}"
MODEL_NAME="${2:?Usage: $0 <model_path> <model_name> [extra_args...]}"
shift 2
EXTRA_ARGS="$*"

PORT="${LLAMA_CPP_PORT:-8080}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"
HEALTH_INTERVAL=5

echo "=== Model path: ${MODEL_PATH} ==="
ls -la "${MODEL_PATH}" || true

echo "=== Starting llama-server ==="
# shellcheck disable=SC2086
llama-server \
  --model "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  ${EXTRA_ARGS} &
SERVER_PID=$!

cleanup() {
  echo "=== Stopping llama-server ==="
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Waiting for health check ==="
elapsed=0
while [ "${elapsed}" -lt "${HEALTH_TIMEOUT}" ]; do
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "Server healthy after ${elapsed}s"
    break
  fi
  sleep "${HEALTH_INTERVAL}"
  elapsed=$((elapsed + HEALTH_INTERVAL))
done

if [ "${elapsed}" -ge "${HEALTH_TIMEOUT}" ]; then
  echo "ERROR: server did not become healthy within ${HEALTH_TIMEOUT}s"
  exit 1
fi

echo "=== Chat completion request ==="
RESPONSE=$(curl -sf "http://localhost:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Reply with a single word: hello\"}],\"max_tokens\":16}")

echo "Response: ${RESPONSE}"

# Assert we got a non-empty assistant message back.
if ! echo "${RESPONSE}" | grep -q '"content"'; then
  echo "ERROR: completion response missing content"
  exit 1
fi

echo "=== SMOKE TEST PASSED (${MODEL_NAME}) ==="
