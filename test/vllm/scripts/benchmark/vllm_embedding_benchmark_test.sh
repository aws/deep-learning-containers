#!/bin/bash
set -euo pipefail

# vLLM Embedding Benchmark Test
# Measures embedding throughput and latency for text and multimodal embedding models.
# Supports Qwen3-Embedding (text) and Qwen3-VL-Embedding (text + image).
#
# Usage: vllm_embedding_benchmark_test.sh <model_dir> <model_name> <runner_type> [extra_vllm_args...]
#
# Environment variables (optional):
# MIN_REQUESTS_PER_SEC - minimum requests/s (default: 5)
# MAX_LATENCY_P99_SEC - maximum p99 latency in seconds (default: 0, disabled)
# NUM_REQUESTS - number of benchmark requests (default: 200)
# CONCURRENCY - number of concurrent requests (default: 8)
# INPUT_LEN - approximate input token length (default: 512)
# BENCHMARK_PROFILES - comma-separated profiles (default: baseline)
#   Profiles: baseline, high_concurrency, batch_scaling

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

echo "=== Embedding Benchmark: ${MODEL_NAME} ==="

pip install -q aiohttp httpx numpy > /dev/null 2>&1

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

export MODEL_DIR MODEL_NAME RESULTS_DIR ARTIFACT_PREFIX VLLM_PORT
export MIN_RPS="${MIN_REQUESTS_PER_SEC:-5}"
export MAX_P99="${MAX_LATENCY_P99_SEC:-0}"
export NUM_REQUESTS="${NUM_REQUESTS:-200}"
export CONCURRENCY="${CONCURRENCY:-8}"
export INPUT_LEN="${INPUT_LEN:-512}"
export BENCHMARK_PROFILES="${BENCHMARK_PROFILES:-baseline}"

echo "=== Running embedding benchmark (profiles: ${BENCHMARK_PROFILES}) ==="
python3 << 'BENCH_EOF'
import asyncio, aiohttp, json, time, statistics, sys, os, random, string

PORT = int(os.environ["VLLM_PORT"])
MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_NAME = os.environ["MODEL_NAME"]
RESULTS_DIR = os.environ["RESULTS_DIR"]
ARTIFACT_PREFIX = os.environ["ARTIFACT_PREFIX"]
INPUT_LEN = int(os.environ["INPUT_LEN"])

# Detect if this is a VL (vision-language) embedding model
IS_VL = "vl" in MODEL_NAME.lower()

# Open-source image URLs for multimodal testing
IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Image_created_with_a_mobile_phone.png/320px-Image_created_with_a_mobile_phone.png",
]

# Profile definitions: (num_requests, concurrency, min_rps, max_p99)
PROFILES = {
    "baseline":         {"num_requests": 200,  "concurrency": 1,   "min_rps": 5,   "max_p99": 0},
    "high_concurrency": {"num_requests": 500,  "concurrency": 32,  "min_rps": 20,  "max_p99": 5.0},
    "batch_scaling":    {"num_requests": 200,  "concurrency": 8,   "min_rps": 10,  "max_p99": 3.0},
}


def generate_random_text(approx_tokens):
    """Generate random text of approximately the given token count (~4 chars/token)."""
    words = []
    chars = 0
    target_chars = approx_tokens * 4
    while chars < target_chars:
        word_len = random.randint(3, 10)
        words.append(''.join(random.choices(string.ascii_lowercase, k=word_len)))
        chars += word_len + 1
    return ' '.join(words)


def make_text_payload(text):
    """Standard /v1/embeddings payload for text-only models."""
    return {"model": MODEL_DIR, "input": [text]}


def make_vl_payload(text, image_url=None):
    """Multimodal payload for VL embedding models via /pooling endpoint.
    Per Qwen3-VL-Embedding docs: system instruction + user content with typed blocks.
    """
    content = []
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    content.append({"type": "text", "text": text})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
        {"role": "user", "content": content},
    ]
    return {"model": MODEL_DIR, "messages": messages}


async def send_request(session, request_id):
    text = generate_random_text(INPUT_LEN)

    if IS_VL:
        # 50% text-only, 50% image+text for VL models
        if request_id % 2 == 0:
            payload = make_vl_payload(text, IMAGE_URLS[request_id % len(IMAGE_URLS)])
        else:
            payload = make_vl_payload(text)
    else:
        payload = make_text_payload(text)

    start = time.perf_counter()
    try:
        endpoint = "/pooling" if IS_VL else "/v1/embeddings"
        async with session.post(
            f"http://localhost:{PORT}{endpoint}",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            result = await resp.json()
            latency = time.perf_counter() - start
        data = result.get("data", [])
        vec_key = "data" if IS_VL else "embedding"
        embedding_dim = len(data[0][vec_key]) if data else 0
        return {"latency": latency, "status": resp.status, "dim": embedding_dim}
    except Exception as e:
        return {"latency": time.perf_counter() - start, "status": 0, "dim": 0, "error": str(e)}


async def run_profile(name, cfg):
    num_requests = cfg["num_requests"]
    concurrency = cfg["concurrency"]
    min_rps = cfg["min_rps"]
    max_p99 = cfg["max_p99"]

    print(f"\n{'='*60}")
    print(f"Profile: {name} | requests={num_requests} concurrency={concurrency}")
    print(f"Model type: {'VL (text+image)' if IS_VL else 'text-only'} | input_len≈{INPUT_LEN} tokens")
    print(f"Thresholds: rps>={min_rps}" + (f" p99<={max_p99}s" if max_p99 > 0 else ""))
    print(f"{'='*60}")

    # Warmup
    print("Warmup (5 requests)...")
    async with aiohttp.ClientSession() as session:
        for i in range(5):
            await send_request(session, i)

    # Benchmark
    sem = asyncio.Semaphore(concurrency)

    async def bounded_request(session, req_id):
        async with sem:
            return await send_request(session, req_id)

    async with aiohttp.ClientSession() as session:
        wall_start = time.perf_counter()
        tasks = [bounded_request(session, i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - wall_start

    successes = [r for r in results if r["status"] == 200 and r["dim"] > 0]
    failures = [r for r in results if r["status"] != 200 or r["dim"] == 0]
    latencies = sorted([r["latency"] for r in successes])
    rps = len(successes) / wall_time if wall_time > 0 else 0

    report = {
        "profile": name,
        "model": MODEL_NAME,
        "model_type": "vl_embedding" if IS_VL else "text_embedding",
        "input_len_approx": INPUT_LEN,
        "num_requests": num_requests,
        "concurrency": concurrency,
        "successful": len(successes),
        "failed": len(failures),
        "wall_time_s": round(wall_time, 2),
        "requests_per_second": round(rps, 3),
        "embeddings_per_second": round(rps, 3),
        "embedding_dim": successes[0]["dim"] if successes else 0,
        "latency_avg_s": round(statistics.mean(latencies), 4) if latencies else 0,
        "latency_p50_s": round(latencies[len(latencies)//2], 4) if latencies else 0,
        "latency_p95_s": round(latencies[int(0.95 * len(latencies))], 4) if latencies else 0,
        "latency_p99_s": round(latencies[int(0.99 * len(latencies))], 4) if latencies else 0,
    }

    with open(f"{RESULTS_DIR}/embedding_benchmark_{ARTIFACT_PREFIX}_{name}.json", "w") as f:
        json.dump(report, f, indent=4)

    print(f"Requests: {report['successful']}/{num_requests} ({report['failed']} failed)")
    print(f"Embedding dim: {report['embedding_dim']}")
    print(f"Wall time: {report['wall_time_s']}s")
    print(f"Embeddings/s: {report['embeddings_per_second']} (min: {min_rps})")
    print(f"Latency avg: {report['latency_avg_s']}s | p50: {report['latency_p50_s']}s | p95: {report['latency_p95_s']}s | p99: {report['latency_p99_s']}s")

    ok = True
    if report["requests_per_second"] < min_rps:
        print(f"FAIL: embeddings/s {report['requests_per_second']} < {min_rps}")
        ok = False
    if max_p99 > 0 and report["latency_p99_s"] > max_p99:
        print(f"FAIL: p99 latency {report['latency_p99_s']}s > {max_p99}s")
        ok = False
    if failures:
        fail_rate = len(failures) / num_requests
        if fail_rate > 0.05:
            print(f"FAIL: error rate {fail_rate:.1%} > 5%")
            for r in failures[:3]:
                err = r.get('error', 'status=' + str(r['status']))
            print(f"  Error: {err}")
            ok = False
        else:
            print(f"WARN: {len(failures)} failed requests ({fail_rate:.1%})")

    if ok:
        print(f"PASS: {name}")
    return name, ok


async def main():
    profile_names = os.environ.get("BENCHMARK_PROFILES", "baseline").split(",")

    # Allow env var override for baseline profile
    if profile_names == ["baseline"]:
        PROFILES["baseline"] = {
            "num_requests": int(os.environ.get("NUM_REQUESTS", "200")),
            "concurrency": int(os.environ.get("CONCURRENCY", "1")),
            "min_rps": float(os.environ.get("MIN_RPS", "5")),
            "max_p99": float(os.environ.get("MAX_P99", "0")),
        }

    all_passed = True
    for pname in profile_names:
        pname = pname.strip()
        if pname not in PROFILES:
            print(f"ERROR: unknown profile '{pname}'. Available: {', '.join(PROFILES.keys())}")
            sys.exit(1)
        _, ok = await run_profile(pname, PROFILES[pname])
        if not ok:
            all_passed = False

    if not all_passed:
        print("\nFAIL: One or more benchmark profiles failed")
        sys.exit(1)
    print("\nPASS: All embedding benchmark profiles passed")

asyncio.run(main())
BENCH_EOF

echo "=== PASSED: ${MODEL_NAME} embedding benchmark ==="
