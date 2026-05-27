#!/bin/bash
set -euo pipefail

# vLLM Reranker Benchmark Test (Qwen3-Reranker)
# Loads the model as a synthetic Qwen3ForSequenceClassification so the standard
# Cohere-compatible /v1/rerank endpoint registers, then measures throughput +
# latency for (query, N-documents) rerank requests.
#
# Usage: vllm_reranker_benchmark_test.sh <model_dir> <model_name> <runner_type> [extra_vllm_args...]
#
# Environment variables (optional):
# MIN_REQUESTS_PER_SEC - minimum requests/s (default: 1)
# MAX_LATENCY_P99_SEC - maximum p99 latency in seconds (default: 0, disabled)
# NUM_REQUESTS - number of benchmark requests (default: 100)
# CONCURRENCY - number of concurrent requests (default: 4)
# DOCS_PER_REQUEST - documents per rerank request (default: 8)
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

CHAT_TEMPLATE="/tmp/qwen3_reranker.jinja"
cat > "${CHAT_TEMPLATE}" <<'JINJA'
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {{ messages | selectattr("role", "eq", "system") | map(attribute="content") | first | default("Given a web search query, retrieve relevant passages that answer the query") }}
<Query>: {{ messages | selectattr("role", "eq", "query") | map(attribute="content") | first }}
<Document>: {{ messages | selectattr("role", "eq", "document") | map(attribute="content") | first }}<|im_end|>
<|im_start|>assistant
<think>

</think>


JINJA

HF_OVERRIDES='{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}'

echo "=== Reranker Benchmark: ${MODEL_NAME} ==="

echo "=== Starting vLLM server (seq-cls Path C) ==="
# shellcheck disable=SC2086
vllm serve "${MODEL_DIR}" \
  --port "${VLLM_PORT}" \
  --runner pooling \
  --hf_overrides "${HF_OVERRIDES}" \
  --chat-template "${CHAT_TEMPLATE}" \
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
export MIN_RPS="${MIN_REQUESTS_PER_SEC:-1}"
export MAX_P99="${MAX_LATENCY_P99_SEC:-0}"
export NUM_REQUESTS="${NUM_REQUESTS:-100}"
export CONCURRENCY="${CONCURRENCY:-4}"
export DOCS_PER_REQUEST="${DOCS_PER_REQUEST:-8}"
export BENCHMARK_PROFILES="${BENCHMARK_PROFILES:-baseline}"

echo "=== Running rerank benchmark (profiles: ${BENCHMARK_PROFILES}) ==="
python3 << 'BENCH_EOF'
import asyncio, aiohttp, json, time, statistics, sys, os, random, string

PORT = int(os.environ["VLLM_PORT"])
MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_NAME = os.environ["MODEL_NAME"]
RESULTS_DIR = os.environ["RESULTS_DIR"]
ARTIFACT_PREFIX = os.environ["ARTIFACT_PREFIX"]
DOCS_PER_REQUEST = int(os.environ["DOCS_PER_REQUEST"])

PROFILES = {
    "baseline":         {"num_requests": 100, "concurrency": 4,  "min_rps": 1,  "max_p99": 0},
    "high_concurrency": {"num_requests": 200, "concurrency": 16, "min_rps": 3,  "max_p99": 30.0},
    "batch_scaling":    {"num_requests": 100, "concurrency": 8,  "min_rps": 2,  "max_p99": 15.0},
}

QUERY_POOL = [
    "What is the capital of France?",
    "How does photosynthesis work?",
    "What is the largest ocean on Earth?",
    "Who wrote the play Hamlet?",
    "What is the speed of light in vacuum?",
]

DOC_POOL = [
    "Paris is the capital and largest city of France.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "The Pacific Ocean is the largest and deepest ocean.",
    "Hamlet was written by William Shakespeare around 1600.",
    "The speed of light in vacuum is approximately 299,792 km/s.",
    "Berlin is the capital of Germany.",
    "The Amazon rainforest is in South America.",
    "Tokyo is the most populous metropolitan area.",
    "Mount Everest is the highest mountain on Earth.",
    "The Sahara is the largest hot desert.",
]


def random_text(approx_chars=200):
    return ' '.join(
        ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 9)))
        for _ in range(approx_chars // 6)
    )


def make_payload(req_id):
    query = QUERY_POOL[req_id % len(QUERY_POOL)]
    # mix real docs + filler so doc list is the requested size
    docs = list(DOC_POOL[: min(DOCS_PER_REQUEST, len(DOC_POOL))])
    while len(docs) < DOCS_PER_REQUEST:
        docs.append(random_text(120))
    random.shuffle(docs)
    return {"model": MODEL_DIR, "query": query, "documents": docs}


async def send_request(session, req_id):
    payload = make_payload(req_id)
    start = time.perf_counter()
    try:
        async with session.post(
            f"http://localhost:{PORT}/v1/rerank",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            result = await resp.json()
            latency = time.perf_counter() - start
        results = result.get("results", [])
        return {
            "latency": latency,
            "status": resp.status,
            "num_results": len(results),
        }
    except Exception as e:
        return {"latency": time.perf_counter() - start, "status": 0, "num_results": 0, "error": str(e)}


async def run_profile(name, cfg):
    num_requests = cfg["num_requests"]
    concurrency = cfg["concurrency"]
    min_rps = cfg["min_rps"]
    max_p99 = cfg["max_p99"]

    print(f"\n{'='*60}")
    print(f"Profile: {name} | requests={num_requests} concurrency={concurrency} docs_per_req={DOCS_PER_REQUEST}")
    print(f"Thresholds: rps>={min_rps}" + (f" p99<={max_p99}s" if max_p99 > 0 else ""))
    print(f"{'='*60}")

    print("Warmup (3 requests)...")
    async with aiohttp.ClientSession() as session:
        for i in range(3):
            await send_request(session, i)

    sem = asyncio.Semaphore(concurrency)

    async def bounded_request(session, req_id):
        async with sem:
            return await send_request(session, req_id)

    async with aiohttp.ClientSession() as session:
        wall_start = time.perf_counter()
        tasks = [bounded_request(session, i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - wall_start

    successes = [r for r in results if r["status"] == 200 and r["num_results"] == DOCS_PER_REQUEST]
    failures = [r for r in results if r["status"] != 200 or r["num_results"] != DOCS_PER_REQUEST]
    latencies = sorted([r["latency"] for r in successes])
    rps = len(successes) / wall_time if wall_time > 0 else 0
    docs_per_sec = rps * DOCS_PER_REQUEST

    report = {
        "profile": name,
        "model": MODEL_NAME,
        "model_type": "reranker",
        "docs_per_request": DOCS_PER_REQUEST,
        "num_requests": num_requests,
        "concurrency": concurrency,
        "successful": len(successes),
        "failed": len(failures),
        "wall_time_s": round(wall_time, 2),
        "requests_per_second": round(rps, 3),
        "documents_per_second": round(docs_per_sec, 3),
        "latency_avg_s": round(statistics.mean(latencies), 4) if latencies else 0,
        "latency_p50_s": round(latencies[len(latencies)//2], 4) if latencies else 0,
        "latency_p95_s": round(latencies[int(0.95 * len(latencies))], 4) if latencies else 0,
        "latency_p99_s": round(latencies[int(0.99 * len(latencies))], 4) if latencies else 0,
    }

    with open(f"{RESULTS_DIR}/reranker_benchmark_{ARTIFACT_PREFIX}_{name}.json", "w") as f:
        json.dump(report, f, indent=4)

    print(f"Requests: {report['successful']}/{num_requests} ({report['failed']} failed)")
    print(f"Wall time: {report['wall_time_s']}s")
    print(f"Requests/s: {report['requests_per_second']} (min: {min_rps})")
    print(f"Documents/s: {report['documents_per_second']}")
    print(f"Latency avg: {report['latency_avg_s']}s | p50: {report['latency_p50_s']}s | p95: {report['latency_p95_s']}s | p99: {report['latency_p99_s']}s")

    ok = True
    if report["requests_per_second"] < min_rps:
        print(f"FAIL: requests/s {report['requests_per_second']} < {min_rps}")
        ok = False
    if max_p99 > 0 and report["latency_p99_s"] > max_p99:
        print(f"FAIL: p99 latency {report['latency_p99_s']}s > {max_p99}s")
        ok = False
    if failures:
        fail_rate = len(failures) / num_requests
        if fail_rate > 0.05:
            print(f"FAIL: error rate {fail_rate:.1%} > 5%")
            for r in failures[:3]:
                err = r.get('error', f"status={r['status']} num_results={r['num_results']}")
                print(f"  Error: {err}")
            ok = False
        else:
            print(f"WARN: {len(failures)} failed requests ({fail_rate:.1%})")

    if ok:
        print(f"PASS: {name}")
    return name, ok


async def main():
    profile_names = os.environ.get("BENCHMARK_PROFILES", "baseline").split(",")

    if profile_names == ["baseline"]:
        PROFILES["baseline"] = {
            "num_requests": int(os.environ.get("NUM_REQUESTS", "100")),
            "concurrency": int(os.environ.get("CONCURRENCY", "4")),
            "min_rps": float(os.environ.get("MIN_RPS", "1")),
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
    print("\nPASS: All reranker benchmark profiles passed")

asyncio.run(main())
BENCH_EOF

echo "=== PASSED: ${MODEL_NAME} reranker benchmark ==="
