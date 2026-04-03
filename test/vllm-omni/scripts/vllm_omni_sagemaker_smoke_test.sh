#!/bin/bash
# Smoke test for vLLM-Omni SageMaker images
# The container is started with the real SageMaker entrypoint (including
# the routing middleware). This script only waits for readiness and tests
# inference via /invocations and /ping — the same path SageMaker uses.
set -eux

MODEL_TYPE="${1:?Usage: $0 <model-type>}"
PORT=8080

echo "=== Testing vLLM-Omni SageMaker: ${MODEL_TYPE} ==="

# Wait for server (entrypoint starts it)
echo "Waiting for server..."
for i in $(seq 1 300); do
    if curl -s http://localhost:${PORT}/ping >/dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 1
done

curl -sf http://localhost:${PORT}/ping || { echo "Ping failed"; exit 1; }
curl -sf http://localhost:${PORT}/health || { echo "Health failed"; exit 1; }

curl -sf http://localhost:${PORT}/v1/models | python3 -c "
import sys, json
data = json.load(sys.stdin)
assert len(data['data']) > 0, 'No models listed'
print(f'Model loaded: {data[\"data\"][0][\"id\"]}')
"

if [ "${MODEL_TYPE}" = "tts" ]; then
    curl -sf -X POST http://localhost:${PORT}/invocations \
      -H "Content-Type: application/json" \
      -H "X-Amzn-SageMaker-Custom-Attributes: route=/v1/audio/speech" \
      -d '{
        "input": "Hello, how are you?",
        "voice": "vivian",
        "language": "English"
      }' --output /tmp/tts_output.wav
    FILE_SIZE=$(stat -c%s /tmp/tts_output.wav 2>/dev/null || stat -f%z /tmp/tts_output.wav)
    echo "TTS output file size: ${FILE_SIZE} bytes"
    [ "${FILE_SIZE}" -gt 1000 ] || { echo "FAIL: TTS output too small"; exit 1; }
    echo "TTS serving test PASSED"

elif [ "${MODEL_TYPE}" = "diffusion" ]; then
    RESPONSE=$(curl -sf http://localhost:${PORT}/invocations \
      -H "Content-Type: application/json" \
      -d '{
        "messages": [{"role": "user", "content": "a red apple on a white table"}],
        "extra_body": {
          "height": 512,
          "width": 512,
          "num_inference_steps": 4,
          "guidance_scale": 3.5,
          "seed": 42
        }
      }')
    echo "${RESPONSE}" | python3 -c "
import sys, json, base64
data = json.load(sys.stdin)
assert 'choices' in data, f'No choices in response: {str(data)[:200]}'
content = data['choices'][0]['message']['content']
if isinstance(content, list):
    img_item = next(c for c in content if c.get('type') == 'image_url')
    url = img_item['image_url']['url']
else:
    url = str(content)
assert 'base64,' in url, 'No base64 image in response'
img_b64 = url.split('base64,')[1]
img_bytes = base64.b64decode(img_b64)
print(f'Image generated: {len(img_bytes)} bytes')
assert len(img_bytes) > 1000, f'Image too small: {len(img_bytes)} bytes'
print('Diffusion serving test PASSED')
"
fi

echo "=== vLLM-Omni SageMaker ${MODEL_TYPE} test PASSED ==="
