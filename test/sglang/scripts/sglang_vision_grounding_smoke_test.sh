#!/bin/bash
set -euo pipefail

# SGLang Vision Grounding Smoke Test
# Tests multimodal models that output bounding boxes (e.g. LocateAnything)
# Usage: sglang_vision_grounding_smoke_test.sh <model_dir> <model_name> [extra_args...]

MODEL_DIR="${1:?Usage: $0 <model_dir> <model_name> [extra_args...]}"
MODEL_NAME="${2:?Usage: $0 <model_dir> <model_name> [extra_args...]}"
shift 2
EXTRA_ARGS="$*"

SGLANG_PORT="${SGLANG_PORT:-30000}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1200}"
HEALTH_INTERVAL=10

echo "=== Model directory: ${MODEL_DIR} ==="
ls -la "${MODEL_DIR}"

echo "=== Starting SGLang server ==="
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

# Generate a 64x64 synthetic image with distinct colored blocks (gives the model
# something to detect — solid single-color images produce no detections).
TEST_IMAGE_B64=$(python3 -c "
import base64, struct, zlib
width, height = 64, 64
raw = b''
for y in range(height):
    raw += b'\x00'  # filter byte
    for x in range(width):
        if x < 32 and y < 32:
            raw += b'\xff\x00\x00'  # red top-left
        elif x >= 32 and y < 32:
            raw += b'\x00\xff\x00'  # green top-right
        elif x < 32 and y >= 32:
            raw += b'\x00\x00\xff'  # blue bottom-left
        else:
            raw += b'\xff\xff\x00'  # yellow bottom-right
ihdr = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
def chunk(ctype, data):
    c = ctype + data
    return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
png = b'\x89PNG\r\n\x1a\n'
png += chunk(b'IHDR', ihdr)
png += chunk(b'IDAT', zlib.compress(raw))
png += chunk(b'IEND', b'')
print(base64.b64encode(png).decode())
")

echo "=== Running vision grounding test ==="
RESPONSE=$(curl -sf http://localhost:${SGLANG_PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_DIR}\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,${TEST_IMAGE_B64}\"}},
        {\"type\": \"text\", \"text\": \"Locate all the instances that matches the following description: red square</c>green square</c>blue square</c>yellow square\"}
      ]
    }],
    \"max_tokens\": 2048
  }")

echo "Response: ${RESPONSE}"

if echo "${RESPONSE}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
choices = data['choices']
assert len(choices) > 0, 'No choices in response'
content = choices[0]['message']['content']
assert len(content) > 0, 'Empty response content'
assert '<box>' in content, f'No bounding box in response: {content}'
print(f'Grounding output: {content}')
"; then
  echo "=== PASSED: ${MODEL_NAME} ==="
else
  echo "=== FAILED: ${MODEL_NAME} - no bounding box in response ==="
  exit 1
fi
