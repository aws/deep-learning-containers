#!/bin/bash
set -euo pipefail

# vLLM ASR Smoke Test (config-driven)
# Follows the vllm-omni pattern: test cases are defined in YAML config and
# passed via TEST_CASES_JSON env var. Each test case specifies a route,
# audio fixture, and validation rule.
#
# Config fields per test case:
#   route:         API endpoint (/v1/chat/completions or /v1/audio/transcriptions)
#   audio_fixture: filename in /models/test-fixtures/
#   validate:      "contains:<text>" or "json_field:<path>"
#
# Audio fixtures are downloaded from S3 by the CI workflow and copied to
# /models/test-fixtures/ before this script runs.
#
# Usage: vllm_asr_smoke_test.sh <model_dir> <model_name> [extra_args...]

MODEL_DIR="${1:?Usage: $0 <model_dir> <model_name> [extra_args...]}"
MODEL_NAME="${2:?Usage: $0 <model_dir> <model_name> [extra_args...]}"
shift 2
EXTRA_ARGS="$*"

VLLM_PORT=8000
HEALTH_TIMEOUT=600
HEALTH_INTERVAL=10
FIXTURES_DIR="/models/test-fixtures"

if [ -z "${TEST_CASES_JSON:-}" ]; then
  echo "ERROR: TEST_CASES_JSON env var not set"
  exit 1
fi

# vllm[audio] (librosa/soundfile) NOT needed — server decodes audio internally via torchaudio/soundfile
# httpx already installed in vLLM container

echo "=== ASR Smoke Test: ${MODEL_NAME} ==="
echo "=== Model directory: ${MODEL_DIR} ==="
echo "=== Test fixtures ==="
ls -lh "${FIXTURES_DIR}"

echo "=== Starting vLLM server ==="
# shellcheck disable=SC2086
vllm serve "${MODEL_DIR}" \
  --port "${VLLM_PORT}" \
  ${EXTRA_ARGS} &
VLLM_PID=$!

cleanup() {
  echo "=== Stopping vLLM server ==="
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
    echo "ERROR: vLLM process died"
    exit 1
  fi
  sleep "${HEALTH_INTERVAL}"
  elapsed=$((elapsed + HEALTH_INTERVAL))
done

if [ "${elapsed}" -ge "${HEALTH_TIMEOUT}" ]; then
  echo "ERROR: Health check timed out after ${HEALTH_TIMEOUT}s"
  exit 1
fi

echo "=== Running test cases ==="
python3 - "${MODEL_DIR}" "${VLLM_PORT}" "${FIXTURES_DIR}" << 'PYEOF'
import httpx, json, sys, base64, os

model_dir, port, fixtures_dir = sys.argv[1], sys.argv[2], sys.argv[3]
test_cases = json.loads(os.environ["TEST_CASES_JSON"])

def load_audio_b64(fixture_name):
    path = f"{fixtures_dir}/{fixture_name}"
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def send_chat_completions(audio_fixture):
    b64 = load_audio_b64(audio_fixture)
    resp = httpx.post(f"http://localhost:{port}/v1/chat/completions", json={
        "model": model_dir,
        "messages": [{"role": "user", "content": [
            {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{b64}"}}
        ]}],
    }, timeout=120)
    return resp.json()

def send_transcription(audio_fixture):
    path = f"{fixtures_dir}/{audio_fixture}"
    with open(path, "rb") as f:
        audio_data = f.read()
    resp = httpx.post(f"http://localhost:{port}/v1/audio/transcriptions",
        files={"file": ("test.wav", audio_data, "audio/wav")},
        data={"model": model_dir}, timeout=120)
    return resp.json()

def validate(result, route, rule):
    """Validate response using omni-style rules."""
    if route == "/v1/chat/completions":
        content = result["choices"][0]["message"]["content"]
        text = content
    elif route == "/v1/audio/transcriptions":
        text = result.get("text", "")
    else:
        raise ValueError(f"Unknown route: {route}")

    if rule.startswith("contains:"):
        expected = rule[len("contains:"):]
        assert expected in text, f"Expected '{expected}' in response, got: {text}"
    elif rule.startswith("json_field:"):
        field = rule[len("json_field:"):]
        obj = result
        for part in field.replace("]", "").replace("[", ".").split("."):
            obj = obj[int(part)] if part.isdigit() else obj[part]
        assert obj, f"Field {field} is empty"
    else:
        raise ValueError(f"Unknown validation rule: {rule}")

    return text

failed = 0
for i, tc in enumerate(test_cases):
    name = tc["name"]
    route = tc["route"]
    audio = tc["audio_fixture"]
    rule = tc["validate"]

    print(f"\n--- Test {i+1}: {name} ---")
    print(f"  Route: {route}, Audio: {audio}, Validate: {rule}")

    try:
        if route == "/v1/chat/completions":
            result = send_chat_completions(audio)
        elif route == "/v1/audio/transcriptions":
            result = send_transcription(audio)
        else:
            raise ValueError(f"Unknown route: {route}")

        text = validate(result, route, rule)
        print(f"  Output: {text[:200]}")
        print(f"  PASS: {name}")
    except Exception as e:
        print(f"  FAIL: {name} — {e}")
        failed += 1

print(f"\n=== Results: {len(test_cases) - failed}/{len(test_cases)} passed ===")
if failed:
    sys.exit(1)
PYEOF

echo "=== PASSED: ${MODEL_NAME} smoke test ==="
