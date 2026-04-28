#!/usr/bin/env bash
# End-to-end TTS example: start server, wait for ready, synthesize speech.
# Requires: docker (with NVIDIA runtime), curl, an authenticated ECR pull.
set -euo pipefail

IMAGE="${IMAGE:-763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm-omni:omni-cuda-v1}"
MODEL="${MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
NAME="${NAME:-omni-tts}"

docker run -d --name "${NAME}" --gpus all --shm-size=2g -p 8080:8000 \
  -v "${HOME}/hf-cache:/root/.cache/huggingface" \
  "${IMAGE}" --model "${MODEL}"

until curl -sf http://localhost:8080/health >/dev/null; do sleep 5; done

curl -sf -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello from vLLM-Omni.", "voice": "vivian", "language": "English"}' \
  --output speech.wav

echo "wrote speech.wav ($(stat -f%z speech.wav 2>/dev/null || stat -c%s speech.wav) bytes)"
# Cleanup:  docker stop "${NAME}" && docker rm "${NAME}"
