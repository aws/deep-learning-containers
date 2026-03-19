#!/bin/bash
set -euo pipefail

# vLLM Model Smoke Test
# Usage: vllm_model_smoke_test.sh <s3_path> <tp> <model_name>

S3_PATH="${1:?Usage: $0 <s3_path> <tp> <model_name>}"
TP="${2:?Usage: $0 <s3_path> <tp> <model_name>}"
MODEL_NAME="${3:?Usage: $0 <s3_path> <tp> <model_name>}"

MODEL_DIR="/tmp/models/${MODEL_NAME}"
VLLM_PORT=8000
HEALTH_TIMEOUT=600
HEALTH_INTERVAL=10

echo "=== Downloading model ${MODEL_NAME} from ${S3_PATH} ==="
mkdir -p "${MODEL_DIR}"
aws s3 cp "${S3_PATH}" /tmp/"${MODEL_NAME}".tar.gz
tar xzf /tmp/"${MODEL_NAME}".tar.gz -C "${MODEL_DIR}"
rm -f /tmp/"${MODEL_NAME}".tar.gz

# If tar created a single subdirectory, use that as MODEL_DIR
SUBDIRS=("${MODEL_DIR}"/*)
if [ ${#SUBDIRS[@]} -eq 1 ] && [ -d "${SUBDIRS[0]}" ]; then
  MODEL_DIR="${SUBDIRS[0]}"
fi

echo "=== Model directory: ${MODEL_DIR} ==="
ls -la "${MODEL_DIR}"

echo "=== Starting vLLM server (tp=${TP}) ==="
vllm serve "${MODEL_DIR}" \
  --tensor-parallel-size "${TP}" \
  --port "${VLLM_PORT}" \
  --disable-log-requests &
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
