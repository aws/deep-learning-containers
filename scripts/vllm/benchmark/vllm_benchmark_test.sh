#!/bin/bash
set -euo pipefail

# vLLM Benchmark Test
# Runs throughput and latency benchmarks using vllm's built-in bench CLI,
# then validates results meet minimum thresholds.
#
# Usage: vllm_benchmark_test.sh <model_dir> <model_name> [extra_vllm_args...]
#
# Environment variables (optional):
# MIN_THROUGHPUT_TOKENS_PER_SEC - minimum output tokens/s (default: 100)
# MIN_REQUESTS_PER_SEC - minimum requests/s (default: 1)
# BENCHMARK_NUM_PROMPTS - number of prompts for throughput test (default: 64)
# BENCHMARK_INPUT_LEN - input token length (default: 512)
# BENCHMARK_OUTPUT_LEN - output token length (default: 128)
# BENCHMARK_BATCH_SIZE - batch size for latency test (default: 4)
# BENCHMARK_LATENCY_ITERS - iterations for latency test (default: 10)

MODEL_DIR="${1:?Usage: $0 <model_dir> <model_name> [extra_vllm_args...]}"
MODEL_NAME="${2:?Usage: $0 <model_dir> <model_name> [extra_vllm_args...]}"
shift 2
EXTRA_ARGS="$*"

RESULTS_DIR="/tmp/benchmark_results"
mkdir -p "${RESULTS_DIR}"

MIN_THROUGHPUT="${MIN_THROUGHPUT_TOKENS_PER_SEC:-100}"
MIN_RPS="${MIN_REQUESTS_PER_SEC:-1}"
NUM_PROMPTS="${BENCHMARK_NUM_PROMPTS:-64}"
INPUT_LEN="${BENCHMARK_INPUT_LEN:-512}"
OUTPUT_LEN="${BENCHMARK_OUTPUT_LEN:-128}"
BATCH_SIZE="${BENCHMARK_BATCH_SIZE:-4}"
LATENCY_ITERS="${BENCHMARK_LATENCY_ITERS:-10}"

echo "=== vLLM Benchmark: ${MODEL_NAME} ==="
echo "Model dir: ${MODEL_DIR}"
echo "Extra args: ${EXTRA_ARGS}"

# --- Throughput benchmark (offline) ---
echo ""
echo "=== Running throughput benchmark ==="
# shellcheck disable=SC2086
vllm bench throughput \
  --model "${MODEL_DIR}" \
  --dataset-name random \
  --random-input-len "${INPUT_LEN}" \
  --random-output-len "${OUTPUT_LEN}" \
  --num-prompts "${NUM_PROMPTS}" \
  --output-json "${RESULTS_DIR}/throughput_${MODEL_NAME}.json" \
  ${EXTRA_ARGS}

echo ""
echo "=== Throughput results ==="
cat "${RESULTS_DIR}/throughput_${MODEL_NAME}.json"

# Validate throughput
python3 -c "
import json, sys
with open('${RESULTS_DIR}/throughput_${MODEL_NAME}.json') as f:
    r = json.load(f)
tps = r['total_num_output_tokens'] / r['elapsed_time']
rps = r['requests_per_second']
print(f'Output tokens/s: {tps:.2f} (min: ${MIN_THROUGHPUT})')
print(f'Requests/s: {rps:.2f} (min: ${MIN_RPS})')
ok = True
if tps < ${MIN_THROUGHPUT}:
    print(f'FAIL: tokens/s {tps:.2f} < ${MIN_THROUGHPUT}')
    ok = False
if rps < ${MIN_RPS}:
    print(f'FAIL: requests/s {rps:.2f} < ${MIN_RPS}')
    ok = False
if not ok:
    sys.exit(1)
print('PASS: throughput thresholds met')
"

# --- Latency benchmark (single batch) ---
echo ""
echo "=== Running latency benchmark ==="
# shellcheck disable=SC2086
vllm bench latency \
  --model "${MODEL_DIR}" \
  --input-len "${INPUT_LEN}" \
  --output-len "${OUTPUT_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --num-iters "${LATENCY_ITERS}" \
  --num-iters-warmup 3 \
  --output-json "${RESULTS_DIR}/latency_${MODEL_NAME}.json" \
  ${EXTRA_ARGS}

echo ""
echo "=== Latency results ==="
python3 -c "
import json
with open('${RESULTS_DIR}/latency_${MODEL_NAME}.json') as f:
    r = json.load(f)
print(f'Avg latency: {r[\"avg_latency\"]:.4f}s')
for k, v in r.get('percentiles', {}).items():
    print(f'p{k} latency: {v:.4f}s')
"

echo ""
echo "=== PASSED: ${MODEL_NAME} benchmark ==="
