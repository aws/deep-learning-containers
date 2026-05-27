#!/bin/bash
set -euo pipefail

# vLLM Reranker Smoke Test (Qwen3-Reranker)
# Loads the model as a synthetic Qwen3ForSequenceClassification so the standard
# Cohere-compatible /v1/rerank endpoint registers, then validates ranking.
#
# Usage: vllm_reranker_smoke_test.sh <model_dir> <model_name> [extra_args...]

MODEL_DIR="${1:?Usage: $0 <model_dir> <model_name> [extra_args...]}"
MODEL_NAME="${2:?Usage: $0 <model_dir> <model_name> [extra_args...]}"
shift 2
EXTRA_ARGS="$*"

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

echo "=== Reranker Smoke Test: ${MODEL_NAME} ==="
echo "=== Model directory: ${MODEL_DIR} ==="

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

echo "=== Running rerank tests ==="
python3 - "${MODEL_DIR}" "${VLLM_PORT}" "${MODEL_NAME}" << 'PYEOF'
import httpx, sys

model_dir, port, model_name = sys.argv[1], sys.argv[2], sys.argv[3]
BASE = f"http://localhost:{port}"

def rerank(query, documents):
    resp = httpx.post(f"{BASE}/v1/rerank",
        json={"model": model_dir, "query": query, "documents": documents},
        timeout=120)
    resp.raise_for_status()
    return resp.json()

# --- Test 1: Basic ranking correctness ---
print("\n--- Test 1: Basic ranking correctness ---")
query = "What is the capital of France?"
docs = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "The Eiffel Tower is in Paris.",
    "Tokyo is the capital of Japan.",
]
body = rerank(query, docs)
results = body["results"]
assert len(results) == len(docs), f"Expected {len(docs)} results, got {len(results)}"
top = results[0]
assert top["index"] == 0, f"Expected Paris (idx 0) on top, got idx {top['index']}"
print(f"  Top doc: {docs[top['index']]} (score={top['relevance_score']:.4f})")
print(f"  PASS: top result is the relevant doc")

# --- Test 2: Score ordering monotonic ---
print("\n--- Test 2: Score ordering monotonic ---")
scores = [r["relevance_score"] for r in results]
assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), \
    f"Scores not sorted descending: {scores}"
print(f"  Scores (sorted desc): {[round(s,4) for s in scores]}")
print(f"  PASS: scores monotonically descending")

# --- Test 3: Relevant beats irrelevant by margin ---
print("\n--- Test 3: Relevant beats irrelevant by margin ---")
by_idx = {r["index"]: r["relevance_score"] for r in results}
relevant = by_idx[0]   # "Paris is the capital of France."
irrelevant = by_idx[3] # "Tokyo is the capital of Japan."
print(f"  relevant={relevant:.4f}  irrelevant={irrelevant:.4f}")
assert relevant > irrelevant + 0.1, \
    f"Relevant should outrank irrelevant by >0.1: {relevant:.4f} vs {irrelevant:.4f}"
print(f"  PASS: relevant doc score >> irrelevant doc score")

# --- Test 4: Single-document request ---
print("\n--- Test 4: Single-document request ---")
single = rerank(query, [docs[0]])
assert len(single["results"]) == 1, "Single-doc request should return 1 result"
print(f"  PASS: single-doc rerank works (score={single['results'][0]['relevance_score']:.4f})")

print(f"\n=== All rerank tests passed for {model_name} ===")
PYEOF

echo "=== PASSED: ${MODEL_NAME} reranker smoke test ==="
