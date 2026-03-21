#!/bin/bash
set -euo pipefail

# vLLM Model Smoke Test
# Usage: vllm_model_smoke_test.sh <model_dir> <model_name> [extra_args...]

MODEL_DIR="${1:?Usage: $0 <model_dir> <model_name> [extra_args...]}"
MODEL_NAME="${2:?Usage: $0 <model_dir> <model_name> [extra_args...]}"
shift 2
EXTRA_ARGS="$*"

VLLM_PORT=8000
HEALTH_TIMEOUT=600
HEALTH_INTERVAL=10

echo "=== Model directory: ${MODEL_DIR} ==="
ls -la "${MODEL_DIR}"

echo "=== Starting vLLM server ==="
# shellcheck disable=SC2086
vllm serve "${MODEL_DIR}" \
  --port "${VLLM_PORT}" \
  ${EXTRA_ARGS} &
VLLM_PID=$!

cleanup() {
  echo "=== Stopping vLLM server ==="
  kill "${VLLM_PID}" 2>/dev/null || true
  wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Waiting for health check ==="
elapsed=0
while [ "${elapsed}" -lt "${HEALTH_TIMEOUT}" ]; do
  if curl -sf http://localhost:${VLLM_PORT}/health >/dev/null 2>&1; then
    echo "Server healthy after ${elapsed}s"
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "ERROR: vLLM process died"
    exit 1
  fi
  sleep "${HEALTH_INTERVAL}"
  elapsed=$((elapsed + HEALTH_INTERVAL))
done

if [ "${elapsed}" -ge "${HEALTH_TIMEOUT}" ]; then
  echo "ERROR: Health check timed out after ${HEALTH_TIMEOUT}s"
  exit 1
fi

echo "=== Running completion test ==="
RESPONSE=$(curl -sf http://localhost:${VLLM_PORT}/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${MODEL_DIR}\", \"prompt\": \"Hello\", \"max_tokens\": 16}")

echo "Response: ${RESPONSE}"

if echo "${RESPONSE}" | python3 -c "import sys,json; c=json.load(sys.stdin)['choices']; assert len(c)>0 and len(c[0]['text'].strip())>0"; then
  echo "=== PASSED: ${MODEL_NAME} ==="
else
  echo "=== FAILED: ${MODEL_NAME} - empty or invalid response ==="
  exit 1
fi
