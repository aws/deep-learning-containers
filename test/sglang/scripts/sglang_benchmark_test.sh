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
echo "=== [DEBUG] Port state BEFORE starting server ==="
echo "--- All python/sglang processes ---"
ps aux | grep -iE "python|sglang" | grep -v grep || echo "No python/sglang processes"
echo "--- Checking port ${SGLANG_PORT} ---"
(ss -tlnp 2>/dev/null || netstat -tlnp 2>/dev/null || cat /proc/net/tcp 2>/dev/null) | grep -i "${SGLANG_PORT}\|$(printf '%X' ${SGLANG_PORT})" || echo "Nothing on ${SGLANG_PORT}"
echo "=== [END DEBUG] ==="

echo ""
echo "=== Killing any process occupying port ${SGLANG_PORT} ==="
python3 -c "
import socket, os, signal, glob, struct

port = ${SGLANG_PORT}
hex_port = '%04X' % port

# Find PIDs using the port via /proc/net/tcp
pids_on_port = set()
try:
    with open('/proc/net/tcp') as f:
        for line in f.readlines()[1:]:
            parts = line.split()
            local_addr = parts[1]
            local_port = int(local_addr.split(':')[1], 16)
            if local_port == port:
                inode = parts[9]
                # Find which PID owns this inode
                for fd_dir in glob.glob('/proc/[0-9]*/fd/*'):
                    try:
                        link = os.readlink(fd_dir)
                        if 'socket:[' + inode + ']' in link:
                            pid = int(fd_dir.split('/')[2])
                            pids_on_port.add(pid)
                    except (OSError, ValueError):
                        pass
except FileNotFoundError:
    pass

if pids_on_port:
    print(f'Found PIDs on port {port}: {pids_on_port}')
    for pid in pids_on_port:
        try:
            cmdline = open(f'/proc/{pid}/cmdline').read().replace('\x00', ' ').strip()
            print(f'  Killing PID {pid}: {cmdline[:120]}')
            os.kill(pid, signal.SIGKILL)
        except (OSError, FileNotFoundError) as e:
            print(f'  PID {pid}: {e}')
else:
    # Verify port is actually free
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(('0.0.0.0', port))
        s.close()
        print(f'Port {port} is free')
    except OSError as e:
        print(f'Port {port} appears occupied but no PID found: {e}')
        # Fallback: kill all sglang-related processes
        os.system('pkill -9 -f sglang 2>/dev/null')
        os.system('pkill -9 -f uvicorn 2>/dev/null')
"
sleep 2
echo "--- After cleanup ---"
ps aux | grep -iE "python|sglang|uvicorn" | grep -v grep || echo "Clean: no stale processes"

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
echo "=== [DEBUG] Port state WHILE SERVING (after health check passes) ==="
echo "--- All python/sglang processes ---"
ps aux | grep -iE "python|sglang|uvicorn" | grep -v grep || echo "No python processes"
echo "--- SGLang PID ${SGLANG_PID} open sockets ---"
ls -la /proc/${SGLANG_PID}/fd 2>/dev/null | head -20 || echo "/proc not available"
cat /proc/${SGLANG_PID}/net/tcp 2>/dev/null | head -10 || true
echo "=== [END DEBUG] ==="

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
echo "=== [DEBUG] Port state BEFORE running benchmark ==="
echo "--- All python/sglang processes ---"
ps aux | grep -iE "python|sglang|uvicorn" | grep -v grep || echo "No python processes"
echo "--- /proc/net/tcp (port ${SGLANG_PORT} = 0x$(printf '%04X' ${SGLANG_PORT})) ---"
cat /proc/net/tcp 2>/dev/null | head -5 || true
echo "=== [END DEBUG] ==="

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
