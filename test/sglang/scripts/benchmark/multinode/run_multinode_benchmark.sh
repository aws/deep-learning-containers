#!/bin/bash
set -euo pipefail

# SGLang Multi-Node Benchmark via LeaderWorkerSet
#
# This script runs FROM a lightweight runner pod and orchestrates:
# 1. Deploy an LWS (leader + worker) serving the model on 2x p5.48xlarge
# 2. Wait for readiness via the ClusterIP Service
# 3. Run bench_serving against the service endpoint
# 4. Validate thresholds
# 5. Tear down the LWS
#
# Usage: run_multinode_benchmark.sh <model_name> <image_uri> [extra_args]
#
# Environment variables:
# MIN_THROUGHPUT_TOKENS_PER_SEC - minimum output tokens/s (default: 100)
# MIN_REQUESTS_PER_SEC - minimum requests/s (default: 1)
# BENCHMARK_NUM_PROMPTS - number of prompts (default: 32)
# BENCHMARK_INPUT_LEN - input token length (default: 512)
# BENCHMARK_OUTPUT_LEN - output token length (default: 128)
# RESULTS_DIR - results output directory (default: /tmp/benchmark_results)
# LWS_MANIFEST - path to LWS YAML (default: auto-detected)
# LWS_NAMESPACE - k8s namespace (default: arc-runners)
# HEALTH_TIMEOUT - max wait for LWS ready in seconds (default: 1200)

MODEL_NAME="${1:?Usage: $0 <model_name> <image_uri> [extra_args]}"
IMAGE_URI="${2:?Usage: $0 <model_name> <image_uri> [extra_args]}"
shift 2
EXTRA_ARGS="${*:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LWS_MANIFEST="${LWS_MANIFEST:-${SCRIPT_DIR}/sglang-deepseek-v4-pro-lws.yaml}"
LWS_NAMESPACE="${LWS_NAMESPACE:-arc-runners}"
LWS_NAME="sglang-deepseek-v4-pro"
SERVICE_NAME="${LWS_NAME}-leader"
SGLANG_PORT=30000
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-2400}"

RESULTS_DIR="${RESULTS_DIR:-/tmp/benchmark_results}"
mkdir -p "${RESULTS_DIR}"

MIN_THROUGHPUT="${MIN_THROUGHPUT_TOKENS_PER_SEC:-100}"
MIN_RPS="${MIN_REQUESTS_PER_SEC:-1}"
NUM_PROMPTS="${BENCHMARK_NUM_PROMPTS:-32}"
INPUT_LEN="${BENCHMARK_INPUT_LEN:-512}"
OUTPUT_LEN="${BENCHMARK_OUTPUT_LEN:-128}"

ARTIFACT_PREFIX="${MODEL_NAME}_multinode"

echo "=== SGLang Multi-Node Benchmark ==="
echo "Model: ${MODEL_NAME}"
echo "Image: ${IMAGE_URI}"
echo "LWS: ${LWS_NAME} (2 nodes, TP=16)"
echo "Thresholds: min_throughput=${MIN_THROUGHPUT}, min_rps=${MIN_RPS}"

cleanup() {
  echo "=== Tearing down LWS ==="
  kubectl delete leaderworkerset "${LWS_NAME}" -n "${LWS_NAMESPACE}" --ignore-not-found --timeout=120s || true
  kubectl delete service "${SERVICE_NAME}" -n "${LWS_NAMESPACE}" --ignore-not-found || true
}
trap cleanup EXIT

# --- Step 1: Deploy LWS ---
echo ""
echo "=== Deploying LeaderWorkerSet ==="

# Substitute image URI and model name into the manifest
sed -e "s|\${SGLANG_IMAGE}|${IMAGE_URI}|g" \
    -e "s|\${MODEL_NAME}|${MODEL_NAME}|g" \
    "${LWS_MANIFEST}" | kubectl apply -n "${LWS_NAMESPACE}" -f -

# --- Step 2: Wait for readiness ---
echo ""
echo "=== Waiting for LWS to become ready (timeout: ${HEALTH_TIMEOUT}s) ==="
elapsed=0
while [ "${elapsed}" -lt "${HEALTH_TIMEOUT}" ]; do
  # Check if leader pod is ready
  READY=$(kubectl get pods -n "${LWS_NAMESPACE}" \
    -l "leaderworkerset.sigs.k8s.io/name=${LWS_NAME},role=leader" \
    -o jsonpath='{.items[0].status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "False")

  if [ "${READY}" = "True" ]; then
    echo "LWS ready after ${elapsed}s"
    break
  fi

  # Print status every 60s
  if [ $((elapsed % 60)) -eq 0 ] && [ "${elapsed}" -gt 0 ]; then
    echo "  [${elapsed}s] Still waiting... Pod status:"
    kubectl get pods -n "${LWS_NAMESPACE}" -l "leaderworkerset.sigs.k8s.io/name=${LWS_NAME}" \
      --no-headers 2>/dev/null || true
  fi

  sleep 10
  elapsed=$((elapsed + 10))
done

if [ "${elapsed}" -ge "${HEALTH_TIMEOUT}" ]; then
  echo "ERROR: LWS health check timed out after ${HEALTH_TIMEOUT}s"
  echo "=== Pod logs (leader, last 50 lines) ==="
  kubectl logs -n "${LWS_NAMESPACE}" \
    -l "leaderworkerset.sigs.k8s.io/name=${LWS_NAME},role=leader" \
    --tail=50 2>/dev/null || true
  exit 1
fi

# Resolve the ClusterIP service endpoint
SERVICE_IP=$(kubectl get service "${SERVICE_NAME}" -n "${LWS_NAMESPACE}" \
  -o jsonpath='{.spec.clusterIP}')
echo "Service endpoint: ${SERVICE_IP}:${SGLANG_PORT}"

# Verify with a health check
if ! curl -sf "http://${SERVICE_IP}:${SGLANG_PORT}/health" >/dev/null 2>&1; then
  echo "ERROR: Service endpoint not responding to health check"
  exit 1
fi
echo "Health check passed"

# --- Step 3: Warmup (trigger DeepGEMM JIT) ---
echo ""
echo "=== Warmup: running mini-benchmark to trigger JIT compilation ==="
python3 -m sglang.bench_serving \
  --backend sglang \
  --host "${SERVICE_IP}" --port "${SGLANG_PORT}" \
  --dataset-name random \
  --random-input-len "${INPUT_LEN}" \
  --random-output-len "${OUTPUT_LEN}" \
  --num-prompts "${NUM_PROMPTS}" \
  --model "/models/${MODEL_NAME}" \
  --output-file /dev/null 2>&1 | tail -5
echo "=== Warmup done, waiting 10s ==="
sleep 10

# --- Step 4: Run benchmark ---
echo ""
echo "=== Running benchmark against ${SERVICE_IP}:${SGLANG_PORT} ==="
echo "num_prompts=${NUM_PROMPTS}, input_len=${INPUT_LEN}, output_len=${OUTPUT_LEN}"

OUTPUT_FILE="${RESULTS_DIR}/throughput_${ARTIFACT_PREFIX}.json"

python3 -m sglang.bench_serving \
  --backend sglang \
  --host "${SERVICE_IP}" --port "${SGLANG_PORT}" \
  --dataset-name random \
  --random-input-len "${INPUT_LEN}" \
  --random-output-len "${OUTPUT_LEN}" \
  --num-prompts "${NUM_PROMPTS}" \
  --model "/models/${MODEL_NAME}" \
  --output-file "${OUTPUT_FILE}" 2>&1 | tee "${RESULTS_DIR}/benchmark_${ARTIFACT_PREFIX}.log"

# --- Step 5: Validate ---
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

print(f'Output tokens/s: {output_tps:.2f} (min: ${MIN_THROUGHPUT})')
print(f'Requests/s: {rps:.2f} (min: ${MIN_RPS})')
print(f'Total time: {total_time:.2f}s')
print(f'Mean TTFT: {mean_ttft:.2f}ms, p99 TTFT: {p99_ttft:.2f}ms')
print(f'Mean TPOT: {mean_tpot:.2f}ms, p99 TPOT: {p99_tpot:.2f}ms')

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
echo "=== PASSED: ${MODEL_NAME} multi-node benchmark ==="
