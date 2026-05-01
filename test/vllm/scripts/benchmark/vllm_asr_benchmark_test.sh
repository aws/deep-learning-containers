#!/bin/bash
set -euo pipefail

# vLLM Qwen3-ASR Benchmark Test
# Measures transcription throughput and latency via /v1/chat/completions with audio_url.
# Reference: https://huggingface.co/Qwen/Qwen3-ASR-1.7B#deployment-with-vllm
#
# Usage: vllm_asr_benchmark_test.sh <model_dir> <model_name> <runner_type> [extra_vllm_args...]
#
# Environment variables (optional):
# MIN_THROUGHPUT_TOKENS_PER_SEC - minimum output tokens/s (default: 50)
# MIN_REQUESTS_PER_SEC - minimum requests/s (default: 1)
# MAX_LATENCY_P99_SEC - maximum p99 latency in seconds (default: 0, disabled)
# NUM_REQUESTS - number of benchmark requests (default: 50)
# CONCURRENCY - number of concurrent requests (default: 1)
# BENCHMARK_PROFILES - comma-separated profiles to run (default: baseline)
#   Profiles: baseline, high_concurrency, sustained_load, burst

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

echo "=== Qwen3-ASR Benchmark: ${MODEL_NAME} ==="

# aiohttp, httpx already installed in vLLM container

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

export MODEL_DIR RESULTS_DIR ARTIFACT_PREFIX VLLM_PORT

# Default single-run env vars (used when BENCHMARK_PROFILES is not set)
export MIN_THROUGHPUT="${MIN_THROUGHPUT_TOKENS_PER_SEC:-50}"
export MIN_RPS="${MIN_REQUESTS_PER_SEC:-1}"
export MAX_P99="${MAX_LATENCY_P99_SEC:-0}"
export NUM_REQUESTS="${NUM_REQUESTS:-50}"
export CONCURRENCY="${CONCURRENCY:-1}"
export BENCHMARK_PROFILES="${BENCHMARK_PROFILES:-baseline}"

echo "=== Running ASR benchmark (profiles: ${BENCHMARK_PROFILES}) ==="
python3 << 'BENCH_EOF'
import asyncio, aiohttp, json, time, statistics, sys, os

PORT = int(os.environ["VLLM_PORT"])
MODEL_DIR = os.environ["MODEL_DIR"]
RESULTS_DIR = os.environ["RESULTS_DIR"]
ARTIFACT_PREFIX = os.environ["ARTIFACT_PREFIX"]

AUDIO_URLS = [
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav",
]

# Profile definitions: (num_requests, concurrency, min_throughput, min_rps, max_p99)
PROFILES = {
    "baseline":         {"num_requests": 50,  "concurrency": 1,  "min_tps": 50,  "min_rps": 1,  "max_p99": 0},
    "high_concurrency": {"num_requests": 200, "concurrency": 32, "min_tps": 100, "min_rps": 5,  "max_p99": 5.0},
    "sustained_load":   {"num_requests": 500, "concurrency": 8,  "min_tps": 80,  "min_rps": 3,  "max_p99": 3.0},
    "burst":            {"num_requests": 50,  "concurrency": 50, "min_tps": 50,  "min_rps": 1,  "max_p99": 10.0},
}

def make_payload(audio_url):
    return {
        "model": MODEL_DIR,
        "messages": [{
            "role": "user",
            "content": [{"type": "audio_url", "audio_url": {"url": audio_url}}],
        }],
    }

async def send_request(session, audio_url):
    payload = make_payload(audio_url)
    start = time.perf_counter()
    try:
        async with session.post(
            f"http://localhost:{PORT}/v1/chat/completions",
            json=payload,
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
    except Exception as e:
        return {"latency": time.perf_counter() - start, "text": "", "status": 0, "completion_tokens": 0, "error": str(e)}

async def run_profile(name, cfg):
    num_requests = cfg["num_requests"]
    concurrency = cfg["concurrency"]
    min_tps = cfg["min_tps"]
    min_rps = cfg["min_rps"]
    max_p99 = cfg["max_p99"]

    print(f"\n{'='*60}")
    print(f"Profile: {name} | requests={num_requests} concurrency={concurrency}")
    print(f"Thresholds: tps>={min_tps} rps>={min_rps}" + (f" p99<={max_p99}s" if max_p99 > 0 else ""))
    print(f"{'='*60}")

    # Warmup
    print("Warmup (3 requests)...")
    async with aiohttp.ClientSession() as session:
        for _ in range(3):
            await send_request(session, AUDIO_URLS[0])

    # Benchmark with concurrency via semaphore
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def bounded_request(session, audio_url):
        async with sem:
            return await send_request(session, audio_url)

    async with aiohttp.ClientSession() as session:
        wall_start = time.perf_counter()
        tasks = [
            bounded_request(session, AUDIO_URLS[i % len(AUDIO_URLS)])
            for i in range(num_requests)
        ]
        results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - wall_start

    successes = [r for r in results if r["status"] == 200 and r["text"]]
    failures = [r for r in results if r["status"] != 200 or not r["text"]]
    latencies = [r["latency"] for r in successes]
    total_tokens = sum(r["completion_tokens"] for r in successes)
    rps = len(successes) / wall_time if wall_time > 0 else 0
    tps = total_tokens / wall_time if wall_time > 0 else 0

    report = {
        "profile": name,
        "model": MODEL_DIR,
        "num_requests": num_requests,
        "concurrency": concurrency,
        "successful": len(successes),
        "failed": len(failures),
        "wall_time_s": round(wall_time, 2),
        "requests_per_second": round(rps, 3),
        "output_tokens_per_second": round(tps, 2),
        "total_completion_tokens": total_tokens,
        "latency_avg_s": round(statistics.mean(latencies), 4) if latencies else 0,
        "latency_p50_s": round(statistics.median(latencies), 4) if latencies else 0,
        "latency_p95_s": round(sorted(latencies)[int(0.95 * len(latencies))], 4) if latencies else 0,
        "latency_p99_s": round(sorted(latencies)[int(0.99 * len(latencies))], 4) if latencies else 0,
    }

    with open(f"{RESULTS_DIR}/asr_benchmark_{ARTIFACT_PREFIX}_{name}.json", "w") as f:
        json.dump(report, f, indent=4)

    print(f"Requests: {report['successful']}/{num_requests} ({report['failed']} failed)")
    print(f"Wall time: {report['wall_time_s']}s")
    print(f"Requests/s: {report['requests_per_second']} (min: {min_rps})")
    print(f"Output tokens/s: {report['output_tokens_per_second']} (min: {min_tps})")
    print(f"Latency avg: {report['latency_avg_s']}s | p50: {report['latency_p50_s']}s | p95: {report['latency_p95_s']}s | p99: {report['latency_p99_s']}s")

    ok = True
    if report["output_tokens_per_second"] < min_tps:
        print(f"FAIL: output tokens/s {report['output_tokens_per_second']} < {min_tps}")
        ok = False
    if report["requests_per_second"] < min_rps:
        print(f"FAIL: requests/s {report['requests_per_second']} < {min_rps}")
        ok = False
    if max_p99 > 0 and report["latency_p99_s"] > max_p99:
        print(f"FAIL: p99 latency {report['latency_p99_s']}s > {max_p99}s")
        ok = False
    if report["failed"] > 0:
        fail_rate = report["failed"] / num_requests
        if fail_rate > 0.05:
            print(f"FAIL: error rate {fail_rate:.1%} > 5%")
            ok = False
        else:
            print(f"WARN: {report['failed']} failed requests ({fail_rate:.1%})")

    return name, ok

async def main():
    profile_names = os.environ.get("BENCHMARK_PROFILES", "baseline").split(",")

    # If only "baseline" with custom env vars, allow override
    if profile_names == ["baseline"]:
        custom_num = int(os.environ.get("NUM_REQUESTS", "50"))
        custom_conc = int(os.environ.get("CONCURRENCY", "1"))
        custom_tps = float(os.environ.get("MIN_THROUGHPUT", "50"))
        custom_rps = float(os.environ.get("MIN_RPS", "1"))
        custom_p99 = float(os.environ.get("MAX_P99", "0"))
        PROFILES["baseline"] = {
            "num_requests": custom_num, "concurrency": custom_conc,
            "min_tps": custom_tps, "min_rps": custom_rps, "max_p99": custom_p99,
        }

    all_passed = True
    for pname in profile_names:
        pname = pname.strip()
        if pname not in PROFILES:
            print(f"ERROR: unknown profile '{pname}'. Available: {', '.join(PROFILES.keys())}")
            sys.exit(1)
        name, ok = await run_profile(pname, PROFILES[pname])
        if not ok:
            all_passed = False

    if not all_passed:
        print("\nFAIL: One or more benchmark profiles failed")
        sys.exit(1)
    print("\nPASS: All benchmark profiles passed")

asyncio.run(main())
BENCH_EOF

echo "=== PASSED: ${MODEL_NAME} ASR benchmark ==="
