#!/usr/bin/env bash
# End-to-end image-generation example: start server, wait for ready, generate.
set -euo pipefail

IMAGE="${IMAGE:-763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm-omni:omni-cuda-v1}"
MODEL="${MODEL:-black-forest-labs/FLUX.2-klein-4B}"
NAME="${NAME:-omni-image}"

docker run -d --name "${NAME}" --gpus all --shm-size=2g -p 8080:8000 \
  -v "${HOME}/hf-cache:/root/.cache/huggingface" \
  "${IMAGE}" --model "${MODEL}"

until curl -sf http://localhost:8080/health >/dev/null; do sleep 5; done

# Response JSON has data[0].b64_json — decode to PNG.
curl -sf -X POST http://localhost:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a red apple on a white table, studio lighting", "size": "512x512", "n": 1}' \
  | python3 -c "import base64,json,sys;open('image.png','wb').write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))"

echo "wrote image.png ($(stat -f%z image.png 2>/dev/null || stat -c%s image.png) bytes)"
# Cleanup:  docker stop "${NAME}" && docker rm "${NAME}"
