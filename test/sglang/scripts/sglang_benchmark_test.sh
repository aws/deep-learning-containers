#!/bin/bash
set -euo pipefail

# SGLang Benchmark Test
# Usage: sglang_benchmark_test.sh <model_dir> <model_name> [extra_args...]
#
# Starts SGLang server, waits for health, runs bench_serving with ShareGPT dataset.

MODEL_DIR="${1:?Usage: $0 <model_dir> <model_name> [extra_args...]}"
MODEL_NAME="${2:?Usage: $0 <model_dir> <model_name> [extra_args...]}"
shift 2
EXTRA_ARGS="$*"

SGLANG_PORT=8000
HEALTH_TIMEOUT=600
HEALTH_INTERVAL=10
DATASET_PATH="/tmp/ShareGPT_V3_unfiltered_cleaned_split.json"

echo "=== Downloading ShareGPT dataset ==="
if [ ! -f "${DATASET_PATH}" ]; then
  wget -q -O "${DATASET_PATH}" \
    https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
fi

echo "=== Starting SGLang server ==="
# shellcheck disable=SC2086
python3 -m sglang.launch_server \
  --model-path "${MODEL_DIR}" \
  --host 0.0.0.0 \
  --port "${SGLANG_PORT}" \
  ${EXTRA_ARGS} &
SGLANG_PID=$!

cleanup() {
  echo "=== Stopping SGLang server ==="
  kill "${SGLANG_PID}" 2>/dev/null || true
  wait "${SGLANG_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Waiting for health check ==="
elapsed=0
while [ "${elapsed}" -lt "${HEALTH_TIMEOUT}" ]; do
  if curl -sf http://localhost:${SGLANG_PORT}/health >/dev/null 2>&1; then
    echo "Server healthy after ${elapsed}s"
    break
  fi
  if ! kill -0 "${SGLANG_PID}" 2>/dev/null; then
    echo "ERROR: SGLang process died"
    exit 1
  fi
  sleep "${HEALTH_INTERVAL}"
  elapsed=$((elapsed + HEALTH_INTERVAL))
done

if [ "${elapsed}" -ge "${HEALTH_TIMEOUT}" ]; then
  echo "ERROR: Health check timed out after ${HEALTH_TIMEOUT}s"
  exit 1
fi

echo "=== Running benchmark: ${MODEL_NAME} ==="
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port "${SGLANG_PORT}" \
  --num-prompts 1000 \
  --model "${MODEL_DIR}" \
  --dataset-name sharegpt \
  --dataset-path "${DATASET_PATH}"

echo "=== PASSED: benchmark ${MODEL_NAME} ==="
