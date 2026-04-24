#!/bin/bash
set -euo pipefail

# vLLM ASR Benchmark Test (config-driven)
# Measures transcription throughput and latency via /v1/chat/completions.
# Audio fixture is specified by benchmark_audio_fixture in the YAML config
# and pre-downloaded to /models/test-fixtures/ by the CI workflow.
#
# Usage: vllm_asr_benchmark_test.sh <model_dir> <model_name> <runner_type> [extra_vllm_args...]
#
# Environment variables (optional):
# BENCHMARK_AUDIO_FIXTURE - audio filename in /models/test-fixtures/ (default: asr_en.wav)
# MIN_THROUGHPUT_TOKENS_PER_SEC - minimum output tokens/s (default: 50)
# MIN_REQUESTS_PER_SEC - minimum requests/s (default: 1)
# NUM_REQUESTS - number of benchmark requests (default: 50)

MODEL_DIR="${1:?Usage: $0 <model_dir> <model_name> <runner_type> [extra_vllm_args...]}"
MODEL_NAME="${2:?Usage: $0 <model_dir> <model_name> <runner_type> [extra_vllm_args...]}"
RUNNER_TYPE="${3:?Usage: $0 <model_dir> <model_name> <runner_type> [extra_vllm_args...]}"
shift 3
EXTRA_ARGS="$*"

ARTIFACT_PREFIX="${MODEL_NAME}_${RUNNER_TYPE}"
RESULTS_DIR="/tmp/benchmark_results"
mkdir -p "${RESULTS_DIR}"

VLLM_PORT=8000
HEALTH_TIMEOUT=600
HEALTH_INTERVAL=10
FIXTURES_DIR="/models/test-fixtures"
AUDIO_FIXTURE="${BENCHMARK_AUDIO_FIXTURE:-asr_en.wav}"

# Install audio dependencies required by vllm for audio processing
pip install -q "vllm[audio]" 2>/dev/null || pip install -q librosa soundfile
pip install -q aiohttp > /dev/null 2>&1

AUDIO_PATH="${FIXTURES_DIR}/${AUDIO_FIXTURE}"
if [ ! -f "${AUDIO_PATH}" ]; then
  echo "ERROR: Missing audio fixture ${AUDIO_PATH}"
  exit 1
fi

echo "=== ASR Benchmark: ${MODEL_NAME} ==="
echo "Audio fixture: ${AUDIO_PATH} ($(stat -c%s "${AUDIO_PATH}" 2>/dev/null || stat -f%z "${AUDIO_PATH}") bytes)"

echo "=== Starting vLLM server ==="
# shellcheck disable=SC2086
vllm serve "${MODEL_DIR}" \
  --port "${VLLM_PORT}" \
  ${EXTRA_ARGS} &
VLLM_PID=$!

cleanup() {
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
    echo "ERROR: vLLM process died"; exit 1
  fi
  sleep "${HEALTH_INTERVAL}"
  elapsed=$((elapsed + HEALTH_INTERVAL))
done
[ "${elapsed}" -ge "${HEALTH_TIMEOUT}" ] && echo "ERROR: timeout" && exit 1

export MODEL_DIR RESULTS_DIR ARTIFACT_PREFIX VLLM_PORT AUDIO_PATH
export MIN_THROUGHPUT="${MIN_THROUGHPUT_TOKENS_PER_SEC:-50}"
export MIN_RPS="${MIN_REQUESTS_PER_SEC:-1}"
export NUM_REQUESTS="${NUM_REQUESTS:-50}"

echo "=== Running ASR benchmark ==="
python3 << 'BENCH_EOF'
import asyncio, aiohttp, json, time, statistics, sys, os, base64

PORT = int(os.environ["VLLM_PORT"])
MODEL_DIR = os.environ["MODEL_DIR"]
NUM_REQUESTS = int(os.environ["NUM_REQUESTS"])
MIN_THROUGHPUT = float(os.environ["MIN_THROUGHPUT"])
MIN_RPS = float(os.environ["MIN_RPS"])
RESULTS_DIR = os.environ["RESULTS_DIR"]
ARTIFACT_PREFIX = os.environ["ARTIFACT_PREFIX"]

# Pre-encode audio as base64 data URI
with open(os.environ["AUDIO_PATH"], "rb") as f:
    AUDIO_DATA_URI = f"data:audio/wav;base64,{base64.b64encode(f.read()).decode()}"

def make_payload():
    return {
        "model": MODEL_DIR,
        "messages": [{
            "role": "user",
            "content": [{"type": "audio_url", "audio_url": {"url": AUDIO_DATA_URI}}],
        }],
    }

async def send_request(session):
    start = time.perf_counter()
    async with session.post(
        f"http://localhost:{PORT}/v1/chat/completions",
        json=make_payload(),
        timeout=aiohttp.ClientTimeout(total=120),
    ) as resp:
        result = await resp.json()
        latency = time.perf_counter() - start
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = result.get("usage", {})
    return {
        "latency": latency,
        "text": content,
        "status": resp.status,
        "completion_tokens": usage.get("completion_tokens", 0),
    }

async def main():
    print("Warmup (3 requests)...")
    async with aiohttp.ClientSession() as session:
        for _ in range(3):
            await send_request(session)

    results = []
    async with aiohttp.ClientSession() as session:
        wall_start = time.perf_counter()
        for _ in range(NUM_REQUESTS):
            results.append(await send_request(session))
        wall_time = time.perf_counter() - wall_start

    successes = [r for r in results if r["status"] == 200 and r["text"]]
    latencies = [r["latency"] for r in successes]
    total_tokens = sum(r["completion_tokens"] for r in successes)
    rps = len(successes) / wall_time
    tps = total_tokens / wall_time if wall_time > 0 else 0

    report = {
        "model": MODEL_DIR,
        "num_requests": NUM_REQUESTS,
        "successful": len(successes),
        "wall_time_s": round(wall_time, 2),
        "requests_per_second": round(rps, 3),
        "output_tokens_per_second": round(tps, 2),
        "total_completion_tokens": total_tokens,
        "latency_avg_s": round(statistics.mean(latencies), 4) if latencies else 0,
        "latency_p50_s": round(statistics.median(latencies), 4) if latencies else 0,
        "latency_p95_s": round(sorted(latencies)[int(0.95 * len(latencies))], 4) if latencies else 0,
        "latency_p99_s": round(sorted(latencies)[int(0.99 * len(latencies))], 4) if latencies else 0,
    }

    with open(f"{RESULTS_DIR}/asr_benchmark_{ARTIFACT_PREFIX}.json", "w") as f:
        json.dump(report, f, indent=4)

    print(f"Requests: {report['successful']}/{NUM_REQUESTS}")
    print(f"Wall time: {report['wall_time_s']}s")
    print(f"Requests/s: {report['requests_per_second']} (min: {MIN_RPS})")
    print(f"Output tokens/s: {report['output_tokens_per_second']} (min: {MIN_THROUGHPUT})")
    print(f"Latency avg/p50/p95/p99: {report['latency_avg_s']}s / {report['latency_p50_s']}s / {report['latency_p95_s']}s / {report['latency_p99_s']}s")

    ok = True
    if report["output_tokens_per_second"] < MIN_THROUGHPUT:
        print(f"FAIL: output tokens/s {report['output_tokens_per_second']} < {MIN_THROUGHPUT}")
        ok = False
    if report["requests_per_second"] < MIN_RPS:
        print(f"FAIL: requests/s {report['requests_per_second']} < {MIN_RPS}")
        ok = False
    if not ok:
        sys.exit(1)
    print("PASS: ASR benchmark thresholds met")

asyncio.run(main())
BENCH_EOF

echo "=== PASSED: ${MODEL_NAME} ASR benchmark ==="
