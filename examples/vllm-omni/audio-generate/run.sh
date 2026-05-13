#!/usr/bin/env bash
# End-to-end audio-generation example: start server, generate a 5-second clip.
# /v1/audio/generate is a diffusion-based text-to-audio endpoint (new in 0.20.0).
# Distinct from /v1/audio/speech (which is TTS — a voice reading words).
set -euo pipefail

IMAGE="${IMAGE:-763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm:omni-cuda-v1}"
MODEL="${MODEL:-stabilityai/stable-audio-open-1.0}"
NAME="${NAME:-omni-audio-generate}"

docker run -d --name "${NAME}" --gpus all --shm-size=2g -p 8080:8000 \
  -v "${HOME}/hf-cache:/root/.cache/huggingface" \
  "${IMAGE}" --model "${MODEL}" --trust-remote-code --enforce-eager

until curl -sf http://localhost:8080/health >/dev/null; do sleep 5; done

curl -sf -X POST http://localhost:8080/v1/audio/generate \
  -H "Content-Type: application/json" \
  -d '{"input": "A jazz piano improvisation", "audio_length": 5.0, "guidance_scale": 7.0, "num_inference_steps": 50, "seed": 42}' \
  --output sound.wav

echo "wrote sound.wav ($(stat -f%z sound.wav 2>/dev/null || stat -c%s sound.wav) bytes)"
# Cleanup:  docker stop "${NAME}" && docker rm "${NAME}"
