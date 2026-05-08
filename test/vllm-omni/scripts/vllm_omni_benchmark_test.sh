#!/bin/bash
# Benchmark dispatcher for vLLM-Omni EC2 images.
#
# Starts the container as a server (via `docker run`, done by the workflow)
# then waits for /health and invokes the correct benchmark client based on
# `benchmark_type`. Parses the resulting JSON and validates thresholds.
#
# Usage from workflow:
#   vllm_omni_benchmark_test.sh <benchmark_type> <config_json>
#
# benchmark_type: tts | tts-base | image | video
# config_json: JSON string with fields per type:
#   tts:       {"concurrency": 4, "num_prompts": 20, "voice": "vivian",
#               "language": "English", "min_rps": 1.5, "min_audio_rtf_mult": 6.0,
#               "max_p95_e2e_ms": 3000}
#   tts-base:  {"concurrency": 4, "num_prompts": 20, "ref_audio_url": "s3://...",
#               "ref_text": "...", "language": "English", "min_rps": 1.3,
#               "min_audio_rtf_mult": 4.5, "max_p95_e2e_ms": 3000}
#   image:     {"concurrency": 1, "num_prompts": 8, "size": "512x512",
#               "min_images_per_s": 0.25, "max_p95_e2e_ms": 3500}
#   video:     {"concurrency": 1, "num_prompts": 6, "num_frames": 17,
#               "num_inference_steps": 4, "size": "480x320",
#               "min_videos_per_s": 0.45, "max_p95_e2e_ms": 2000}
set -euo pipefail

BENCHMARK_TYPE="${1:?Usage: $0 <tts|tts-base|image|video> <config_json>}"
CONFIG_JSON="${2:?Usage: $0 <tts|tts-base|image|video> <config_json>}"

PORT="${VLLM_PORT:-8080}"
BASE_URL="http://localhost:${PORT}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/benchmark_results}"
ARTIFACT_PREFIX="${ARTIFACT_PREFIX:-omni-bench}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${RESULTS_DIR}"

echo "=== vLLM-Omni benchmark: ${BENCHMARK_TYPE} ==="
echo "Config: ${CONFIG_JSON}"
echo "Results dir: ${RESULTS_DIR}"

# --- Wait for server ---
echo ""
echo "=== Waiting for /health (up to 600s) ==="
for i in $(seq 1 600); do
  if curl -sf "${BASE_URL}/health" >/dev/null 2>&1; then
    echo "Server ready after ${i}s"
    break
  fi
  sleep 1
done
curl -sf "${BASE_URL}/health" >/dev/null || { echo "Health check failed"; exit 1; }

# --- Install aiohttp if missing (CodeBuild host may or may not have it) ---
python3 -c 'import aiohttp' 2>/dev/null || python3 -m pip install --user --quiet aiohttp

# --- Config extraction helper ---
get_cfg() {
  python3 -c "import json,sys; d=json.loads(sys.argv[1]); v=d.get(sys.argv[2], sys.argv[3] if len(sys.argv)>3 else ''); print(v if v != '' else '')" "${CONFIG_JSON}" "$1" "${2:-}"
}

RESULT_JSON="${RESULTS_DIR}/${ARTIFACT_PREFIX}.json"

# --- Dispatch ---
case "${BENCHMARK_TYPE}" in
  tts)
    CONCURRENCY="$(get_cfg concurrency 4)"
    NUM_PROMPTS="$(get_cfg num_prompts 20)"
    VOICE="$(get_cfg voice vivian)"
    LANGUAGE="$(get_cfg language English)"
    python3 "${SCRIPT_DIR}/benchmark/tts_benchmark_client.py" \
      --base-url "${BASE_URL}" \
      --num-prompts "${NUM_PROMPTS}" --concurrency "${CONCURRENCY}" --warmup 2 \
      --voice "${VOICE}" --language "${LANGUAGE}" \
      --output-json "${RESULT_JSON}"
    ;;

  tts-base)
    CONCURRENCY="$(get_cfg concurrency 4)"
    NUM_PROMPTS="$(get_cfg num_prompts 20)"
    REF_AUDIO_S3="$(get_cfg ref_audio_s3)"
    LANGUAGE="$(get_cfg language English)"
    REF_LOCAL=/tmp/ref_audio.wav
    REF_TEXT_LOCAL=/tmp/ref_audio.txt
    if [ -n "${REF_AUDIO_S3}" ]; then
      aws s3 cp "${REF_AUDIO_S3}" "${REF_LOCAL}"
      # Download transcript (.txt next to .wav) — ref_text MUST match the audio exactly.
      # See: https://github.com/vllm-project/vllm-omni/issues/3124
      REF_TEXT_S3="${REF_AUDIO_S3%.wav}.txt"
      if aws s3 cp "${REF_TEXT_S3}" "${REF_TEXT_LOCAL}" 2>/dev/null; then
        REF_TEXT="$(cat "${REF_TEXT_LOCAL}" | tr -d '\n')"
      else
        REF_TEXT="$(get_cfg ref_text)"
        if [ -z "${REF_TEXT}" ]; then
          echo "ERROR: tts-base requires ref_text in config or a .txt file next to the .wav in S3"
          exit 1
        fi
      fi
    else
      echo "ERROR: tts-base requires ref_audio_s3 in config"
      exit 1
    fi
    echo "ref_text='${REF_TEXT}'"
    python3 "${SCRIPT_DIR}/benchmark/tts_benchmark_client.py" \
      --base-url "${BASE_URL}" \
      --num-prompts "${NUM_PROMPTS}" --concurrency "${CONCURRENCY}" --warmup 2 \
      --task-type Base --ref-audio "${REF_LOCAL}" --ref-text "${REF_TEXT}" \
      --language "${LANGUAGE}" \
      --output-json "${RESULT_JSON}"
    ;;

  image)
    CONCURRENCY="$(get_cfg concurrency 1)"
    NUM_PROMPTS="$(get_cfg num_prompts 8)"
    SIZE="$(get_cfg size 512x512)"
    STEPS="$(get_cfg num_inference_steps)"
    EXTRA=""
    [ -n "${STEPS}" ] && EXTRA="--num-inference-steps ${STEPS}"
    python3 "${SCRIPT_DIR}/benchmark/image_benchmark_client.py" \
      --base-url "${BASE_URL}" \
      --num-prompts "${NUM_PROMPTS}" --concurrency "${CONCURRENCY}" --warmup 1 \
      --size "${SIZE}" ${EXTRA} \
      --output-json "${RESULT_JSON}"
    ;;

  video)
    CONCURRENCY="$(get_cfg concurrency 1)"
    NUM_PROMPTS="$(get_cfg num_prompts 6)"
    NUM_FRAMES="$(get_cfg num_frames 17)"
    STEPS="$(get_cfg num_inference_steps 4)"
    SIZE="$(get_cfg size 480x320)"
    python3 "${SCRIPT_DIR}/benchmark/video_benchmark_client.py" \
      --base-url "${BASE_URL}" \
      --num-prompts "${NUM_PROMPTS}" --concurrency "${CONCURRENCY}" --warmup 1 \
      --num-frames "${NUM_FRAMES}" --num-inference-steps "${STEPS}" --size "${SIZE}" \
      --output-json "${RESULT_JSON}"
    ;;

  chat)
    CONCURRENCY="$(get_cfg concurrency 2)"
    NUM_PROMPTS="$(get_cfg num_prompts 16)"
    MAX_TOKENS="$(get_cfg max_tokens 128)"
    IGNORE_EOS="$(get_cfg ignore_eos)"
    MODEL="$(get_cfg model)"
    EXTRA=""
    [ "${IGNORE_EOS}" = "true" ] && EXTRA="--ignore-eos"
    [ -n "${MODEL}" ] && EXTRA="${EXTRA} --model ${MODEL}"
    python3 "${SCRIPT_DIR}/benchmark/chat_omni_benchmark_client.py" \
      --base-url "${BASE_URL}" \
      --num-prompts "${NUM_PROMPTS}" --concurrency "${CONCURRENCY}" --warmup 2 \
      --max-tokens "${MAX_TOKENS}" ${EXTRA} \
      --output-json "${RESULT_JSON}"
    ;;

  *)
    echo "Unknown benchmark_type: ${BENCHMARK_TYPE}"
    exit 1
    ;;
esac

# --- Validate thresholds (all optional; missing fields skip check) ---
echo ""
echo "=== Validate thresholds ==="
python3 - "${RESULT_JSON}" "${CONFIG_JSON}" "${BENCHMARK_TYPE}" <<'PY'
import json, sys
result_path, cfg_json, btype = sys.argv[1], sys.argv[2], sys.argv[3]
with open(result_path) as f:
    summary = json.load(f).get("summary", {})
cfg = json.loads(cfg_json)

checks = []  # (name, actual, op, expected)
def need(field):
    return cfg.get(field)

# Basic: all requests succeeded
if summary.get("failed", 0) > 0:
    print(f"FAIL: {summary['failed']} requests failed")
    print(json.dumps(summary.get("failure_samples", []), indent=2))
    sys.exit(1)

if btype in ("tts", "tts-base"):
    if need("min_rps"):
        checks.append(("requests_per_second", summary.get("requests_per_second", 0), ">=", cfg["min_rps"]))
    if need("min_audio_rtf_mult"):
        checks.append(("audio_throughput_s_per_s", summary.get("audio_throughput_s_per_s", 0), ">=", cfg["min_audio_rtf_mult"]))
    if need("max_p95_e2e_ms"):
        checks.append(("e2e_ms.p95", summary.get("e2e_ms", {}).get("p95", 0), "<=", cfg["max_p95_e2e_ms"]))

elif btype == "image":
    if need("min_images_per_s"):
        checks.append(("images_per_second", summary.get("images_per_second", 0), ">=", cfg["min_images_per_s"]))
    if need("max_p95_e2e_ms"):
        checks.append(("e2e_ms.p95", summary.get("e2e_ms", {}).get("p95", 0), "<=", cfg["max_p95_e2e_ms"]))

elif btype == "video":
    if need("min_videos_per_s"):
        checks.append(("videos_per_second", summary.get("videos_per_second", 0), ">=", cfg["min_videos_per_s"]))
    if need("max_p95_e2e_ms"):
        checks.append(("e2e_ms.p95", summary.get("e2e_ms", {}).get("p95", 0), "<=", cfg["max_p95_e2e_ms"]))

elif btype == "chat":
    if need("min_rps"):
        checks.append(("requests_per_second", summary.get("requests_per_second", 0), ">=", cfg["min_rps"]))
    if need("min_output_tps"):
        checks.append(("output_tokens_per_second", summary.get("output_tokens_per_second", 0), ">=", cfg["min_output_tps"]))
    if need("max_p95_ttft_ms"):
        checks.append(("ttft_ms.p95", summary.get("ttft_ms", {}).get("p95", 0), "<=", cfg["max_p95_ttft_ms"]))
    if need("max_p95_tpot_ms"):
        checks.append(("tpot_ms.p95", summary.get("tpot_ms", {}).get("p95", 0), "<=", cfg["max_p95_tpot_ms"]))
    if need("max_p95_e2e_ms"):
        checks.append(("e2e_ms.p95", summary.get("e2e_ms", {}).get("p95", 0), "<=", cfg["max_p95_e2e_ms"]))

ok = True
for name, actual, op, expected in checks:
    a = float(actual or 0)
    e = float(expected)
    passed = (a >= e) if op == ">=" else (a <= e)
    status = "PASS" if passed else "FAIL"
    print(f"{status}: {name}={a} {op} {e}")
    if not passed:
        ok = False

# Write threshold result back into the JSON so the report can read it.
with open(result_path) as f:
    doc = json.load(f)
doc.setdefault("summary", {})["threshold_passed"] = ok
with open(result_path, "w") as f:
    json.dump(doc, f, indent=2)

if not ok:
    sys.exit(1)
print("ALL THRESHOLDS PASSED")
PY

echo ""
echo "=== vLLM-Omni benchmark PASSED ==="
