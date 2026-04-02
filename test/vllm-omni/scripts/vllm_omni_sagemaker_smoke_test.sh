#!/bin/bash
# Smoke test for vLLM-Omni SageMaker images
# Validates the server starts with --omni and responds to requests
set -eux

nvidia-smi

MODEL_PATH="${1:?Usage: $0 <model-path> <model-type>}"
MODEL_TYPE="${2:?Usage: $0 <model-path> <model-type>}"
PORT=8091

echo "=== Testing vLLM-Omni SageMaker: ${MODEL_TYPE} at ${MODEL_PATH} ==="

# Start server in background
vllm serve --omni --model "${MODEL_PATH}" --port ${PORT} --enforce-eager --stage-init-timeout 600 &
SERVER_PID=$!

cleanup() {
    echo "Stopping server (PID ${SERVER_PID})..."
    kill ${SERVER_PID} 2>/dev/null || true
    wait ${SERVER_PID} 2>/dev/null || true
}
trap cleanup EXIT

# Wait for server to be ready
echo "Waiting for server to start..."
for i in $(seq 1 300); do
    if curl -s http://localhost:${PORT}/health >/dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "ERROR: Server process died"
        exit 1
    fi
    sleep 1
done

# Verify health endpoint
curl -sf http://localhost:${PORT}/health || { echo "Health check failed"; exit 1; }

# Verify models endpoint
curl -sf http://localhost:${PORT}/v1/models | python3 -c "
import sys, json
data = json.load(sys.stdin)
assert len(data['data']) > 0, 'No models listed'
print(f'Model loaded: {data[\"data\"][0][\"id\"]}')
"

if [ "${MODEL_TYPE}" = "tts" ]; then
    # TTS via /v1/audio/speech API (OpenAI-compatible speech endpoint)
    curl -sf -X POST http://localhost:${PORT}/v1/audio/speech \
      -H "Content-Type: application/json" \
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
    # Image generation via chat completions API
    RESPONSE=$(curl -sf http://localhost:${PORT}/v1/chat/completions \
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
import sys, json
data = json.load(sys.stdin)
assert 'choices' in data, 'No choices in response'
content = data['choices'][0]['message']['content']
print(f'Image generation response received, content type: {type(content)}')
print('Diffusion serving test PASSED')
"
fi

echo "=== vLLM-Omni SageMaker ${MODEL_TYPE} test PASSED ==="
