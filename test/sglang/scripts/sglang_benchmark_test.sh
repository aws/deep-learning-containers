#!/bin/bash
set -euo pipefail

# SGLang Benchmark Test
# Runs online serving benchmark using sglang.bench_serving with threshold validation.
#
# Usage: sglang_benchmark_test.sh <model_dir> <model_name> <runner_type> [extra_sglang_args...]
#
# Environment variables (optional):
# MIN_THROUGHPUT_TOKENS_PER_SEC - minimum output tokens/s (default: 100)
# MIN_REQUESTS_PER_SEC - minimum requests/s (default: 1)
# BENCHMARK_NUM_PROMPTS - number of prompts (default: 64)
# BENCHMARK_INPUT_LEN - input token length (default: 512)
# BENCHMARK_OUTPUT_LEN - output token length (default: 128)
# SGLANG_ENV_VARS - space-separated env vars for server (e.g., "SGLANG_DSV4_FP4_EXPERTS=0")
# RESULTS_DIR - directory for JSON results (default: /tmp/benchmark_results)
# SGLANG_PORT - server port (default: 8000)

MODEL_DIR="${1:?Usage: $0 <model_dir> <model_name> <runner_type> [extra_sglang_args...]}"
MODEL_NAME="${2:?Usage: $0 <model_dir> <model_name> <runner_type> [extra_sglang_args...]}"
RUNNER_TYPE="${3:?Usage: $0 <model_dir> <model_name> <runner_type> [extra_sglang_args...]}"
shift 3
EXTRA_ARGS="$*"

ARTIFACT_PREFIX="${MODEL_NAME}_${RUNNER_TYPE}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/benchmark_results}"
mkdir -p "${RESULTS_DIR}"

if [ -n "${SGLANG_ENV_VARS:-}" ]; then
  echo "=== Setting server env vars ==="
  for var in ${SGLANG_ENV_VARS}; do
    export "${var}"
    echo "  export ${var}"
  done
fi


SGLANG_PORT="${SGLANG_PORT:-30000}"
SGLANG_PORT=$((SGLANG_PORT + RANDOM % 1000))
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1200}"
HEALTH_INTERVAL=10
MIN_THROUGHPUT="${MIN_THROUGHPUT_TOKENS_PER_SEC:-100}"
MIN_RPS="${MIN_REQUESTS_PER_SEC:-1}"
NUM_PROMPTS="${BENCHMARK_NUM_PROMPTS:-64}"
INPUT_LEN="${BENCHMARK_INPUT_LEN:-512}"
OUTPUT_LEN="${BENCHMARK_OUTPUT_LEN:-128}"

echo "=== SGLang Benchmark: ${MODEL_NAME} ==="
echo "Model dir: ${MODEL_DIR}"
echo "Runner: ${RUNNER_TYPE}"
echo "Extra args: ${EXTRA_ARGS}"
echo "Thresholds: min_throughput=${MIN_THROUGHPUT} tok/s, min_rps=${MIN_RPS} req/s"

echo ""
echo "=== Starting SGLang server on port ${SGLANG_PORT} ==="
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

echo ""
echo "=== Warmup: running mini-benchmark to trigger JIT compilation ==="
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port "${SGLANG_PORT}" \
  --dataset-name random \
  --random-input-len "${INPUT_LEN}" \
  --random-output-len "${OUTPUT_LEN}" \
  --num-prompts "${NUM_PROMPTS}" \
  --model "${MODEL_DIR}" \
  --output-file /dev/null 2>&1 | tail -5
echo "=== Warmup benchmark done, waiting 10s ==="
sleep 10

OUTPUT_FILE="${RESULTS_DIR}/throughput_${ARTIFACT_PREFIX}.json"

echo ""
echo "=== Running benchmark (random dataset) ==="
echo "num_prompts=${NUM_PROMPTS}, input_len=${INPUT_LEN}, output_len=${OUTPUT_LEN}"

python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port "${SGLANG_PORT}" \
  --dataset-name random \
  --random-input-len "${INPUT_LEN}" \
  --random-output-len "${OUTPUT_LEN}" \
  --num-prompts "${NUM_PROMPTS}" \
  --model "${MODEL_DIR}" \
  --output-file "${OUTPUT_FILE}" 2>&1 | tee "${RESULTS_DIR}/benchmark_${ARTIFACT_PREFIX}.log"

echo ""
echo "=== Validating results ==="
python3 -c "
import json, sys

with open('${OUTPUT_FILE}') as f:
    r = json.load(f)

output_tps = r.get('output_throughput', 0)
rps = r.get('request_throughput', 0)
total_time = r.get('total_time', 0)
mean_ttft = r.get('mean_ttft_ms', 0)
p99_ttft = r.get('p99_ttft_ms', 0)
mean_tpot = r.get('mean_tpot_ms', 0)
p99_tpot = r.get('p99_tpot_ms', 0)
mean_itl = r.get('mean_itl_ms', 0)
p99_itl = r.get('p99_itl_ms', 0)

print(f'Output tokens/s: {output_tps:.2f} (min: ${MIN_THROUGHPUT})')
print(f'Requests/s: {rps:.2f} (min: ${MIN_RPS})')
print(f'Total time: {total_time:.2f}s')
print(f'Mean TTFT: {mean_ttft:.2f}ms, p99 TTFT: {p99_ttft:.2f}ms')
print(f'Mean TPOT: {mean_tpot:.2f}ms, p99 TPOT: {p99_tpot:.2f}ms')
print(f'Mean ITL: {mean_itl:.2f}ms, p99 ITL: {p99_itl:.2f}ms')

ok = True
if output_tps < ${MIN_THROUGHPUT}:
    print(f'FAIL: output tokens/s {output_tps:.2f} < ${MIN_THROUGHPUT}')
    ok = False
if rps < ${MIN_RPS}:
    print(f'FAIL: requests/s {rps:.2f} < ${MIN_RPS}')
    ok = False
if not ok:
    sys.exit(1)
print('PASS: thresholds met')
"

echo ""
echo "=== PASSED: ${MODEL_NAME} benchmark ==="
