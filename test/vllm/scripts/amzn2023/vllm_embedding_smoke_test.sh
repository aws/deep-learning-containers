#!/bin/bash
set -euo pipefail

# vLLM Embedding Smoke Test
# Tests Qwen3-Embedding (text) and Qwen3-VL-Embedding (multimodal) models.
# Validates: model loads, embeddings returned, cosine similarity ordering correct
#
# Usage: vllm_embedding_smoke_test.sh <model_dir> <model_name> [extra_args...]

MODEL_DIR="${1:?Usage: $0 <model_dir> <model_name> [extra_args...]}"
MODEL_NAME="${2:?Usage: $0 <model_dir> <model_name> [extra_args...]}"
shift 2
EXTRA_ARGS="$*"

VLLM_PORT=8000
HEALTH_TIMEOUT=600
HEALTH_INTERVAL=10

# httpx, numpy already installed in vLLM container

echo "=== Embedding Smoke Test: ${MODEL_NAME} ==="
echo "=== Model directory: ${MODEL_DIR} ==="

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

echo "=== Running embedding tests ==="
python3 - "${MODEL_DIR}" "${VLLM_PORT}" "${MODEL_NAME}" << 'PYEOF'
import httpx, json, sys, numpy as np, time

model_dir, port, model_name = sys.argv[1], sys.argv[2], sys.argv[3]
BASE = f"http://localhost:{port}"
IS_VL = "vl" in model_name.lower()

def get_embeddings(inputs):
    """Call embedding endpoint. Uses /pooling with chat messages for VL models."""
    if IS_VL:
        # VL model: each input is a separate /pooling request with chat format
        # Per Qwen3-VL-Embedding docs: system instruction + user content
        results = []
        for text in inputs:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
                {"role": "user", "content": [{"type": "text", "text": text}]},
            ]
            resp = httpx.post(f"{BASE}/pooling",
                json={"model": model_dir, "messages": messages},
                timeout=120)
            resp.raise_for_status()
            data = resp.json()["data"]
            results.append(np.array(data[0]["data"]))
        return results
    else:
        resp = httpx.post(f"{BASE}/v1/embeddings",
            json={"model": model_dir, "input": inputs},
            timeout=120)
        resp.raise_for_status()
        data = resp.json()["data"]
        return [np.array(d["embedding"]) for d in data]

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# --- Test 1: Basic embedding generation ---
print("\n--- Test 1: Basic embedding generation ---")
texts = ["The capital of France is Paris.", "Machine learning is a subset of AI."]
embeddings = get_embeddings(texts)
assert len(embeddings) == 2, f"Expected 2 embeddings, got {len(embeddings)}"
dim = len(embeddings[0])
print(f"  Embedding dimension: {dim}")
assert dim > 0, "Embedding dimension is 0"
# Qwen3-Embedding-0.6B: 1024, Qwen3-VL-Embedding-2B: 2048
assert dim in (1024, 2048, 768, 512, 256), f"Unexpected dimension: {dim}"
print(f"  PASS: Generated {len(embeddings)} embeddings of dim {dim}")

# --- Test 2: Cosine similarity ordering ---
print("\n--- Test 2: Cosine similarity ordering ---")
query = "What is the capital of France?"
doc_relevant = "Paris is the capital and largest city of France."
doc_irrelevant = "The Amazon rainforest covers most of northwestern Brazil."

embs = get_embeddings([query, doc_relevant, doc_irrelevant])
sim_relevant = cosine_sim(embs[0], embs[1])
sim_irrelevant = cosine_sim(embs[0], embs[2])
print(f"  Similarity (query, relevant): {sim_relevant:.4f}")
print(f"  Similarity (query, irrelevant): {sim_irrelevant:.4f}")
assert sim_relevant > sim_irrelevant, \
    f"Relevant doc should be more similar: {sim_relevant:.4f} <= {sim_irrelevant:.4f}"
print(f"  PASS: Correct similarity ordering")

# --- Test 3: Batch consistency ---
print("\n--- Test 3: Batch consistency ---")
single = get_embeddings(["Hello world"])[0]
batch = get_embeddings(["Hello world", "Another text"])[0]
sim = cosine_sim(single, batch)
print(f"  Single vs batch similarity: {sim:.6f}")
assert sim > 0.99, f"Batch inconsistency: similarity {sim:.6f} < 0.99"
print(f"  PASS: Batch-consistent embeddings")

# --- Test 4: Throughput baseline ---
print("\n--- Test 4: Throughput baseline ---")
texts_batch = [f"This is test sentence number {i} for throughput measurement." for i in range(32)]
start = time.perf_counter()
embs = get_embeddings(texts_batch)
elapsed = time.perf_counter() - start
throughput = len(texts_batch) / elapsed
print(f"  32 embeddings in {elapsed:.2f}s = {throughput:.1f} embeddings/s")
assert len(embs) == 32, f"Expected 32 embeddings, got {len(embs)}"
assert throughput > 1.0, f"Throughput too low: {throughput:.1f} emb/s"
print(f"  PASS: Throughput {throughput:.1f} emb/s")

# --- Test 5: Instruction-aware (Qwen3-Embedding specific) ---
print("\n--- Test 5: Instruction-aware embedding ---")
task_instruction = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
query_with_inst = task_instruction + "What is gravity?"
query_without_inst = "What is gravity?"
doc = "Gravity is a force that attracts two bodies towards each other."

embs = get_embeddings([query_with_inst, query_without_inst, doc])
sim_with = cosine_sim(embs[0], embs[2])
sim_without = cosine_sim(embs[1], embs[2])
print(f"  With instruction: {sim_with:.4f}")
print(f"  Without instruction: {sim_without:.4f}")
# Both should produce reasonable similarity; instruction may or may not improve
assert sim_with > 0.3, f"Instruction-aware similarity too low: {sim_with:.4f}"
print(f"  PASS: Instruction-aware embedding works")

print(f"\n=== All tests passed for {model_name} ===")
PYEOF

echo "=== PASSED: ${MODEL_NAME} embedding smoke test ==="
