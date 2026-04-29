#!/usr/bin/env bash
# End-to-end Qwen2.5-Omni-3B example: start server, wait for ready,
# generate speech via /v1/chat/completions.
#
# REQUIRES ≥ 4 GPUs (e.g., g5.12xlarge / g6.12xlarge / g6e.12xlarge).
# On single-GPU hosts the model's talker stage fails to load on GPU 1.
set -euo pipefail

IMAGE="${IMAGE:-763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm-omni:omni-cuda-v1}"
MODEL="${MODEL:-Qwen/Qwen2.5-Omni-3B}"
NAME="${NAME:-omni3b}"

docker run -d --name "${NAME}" --gpus all --shm-size=16g -p 8080:8080 \
  -v "${HOME}/hf-cache:/root/.cache/huggingface" \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  "${IMAGE}" --model "${MODEL}" \
  --host 0.0.0.0 --port 8080 \
  --max-model-len 16384 --dtype bfloat16

# First start takes ~8 min (weight download + 3-stage load).
until curl -sf http://localhost:8080/health >/dev/null; do sleep 10; done

# Three things are REQUIRED for clean audio:
#   1. "modalities": ["audio"]  (NOT ["text","audio"] — returns empty audio)
#   2. "sampling_params_list"   (3-element list: thinker, talker, code2wav;
#                                built-in defaults produce noise)
#   3. The exact Qwen system prompt below.
# Omitting #2 returns 200 OK with valid WAV bytes that sound like noise.
curl -sf -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-Omni-3B",
    "modalities": ["audio"],
    "sampling_params_list": [
      {"temperature":0.0,"top_p":1.0,"top_k":-1,"max_tokens":2048,"seed":42,"detokenize":true,"repetition_penalty":1.1},
      {"temperature":0.9,"top_p":0.8,"top_k":40,"max_tokens":2048,"seed":42,"detokenize":true,"repetition_penalty":1.05,"stop_token_ids":[8294]},
      {"temperature":0.0,"top_p":1.0,"top_k":-1,"max_tokens":2048,"seed":42,"detokenize":true,"repetition_penalty":1.1}
    ],
    "messages": [
      {"role":"system","content":[{"type":"text","text":"You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
      {"role":"user","content":[{"type":"text","text":"Tell me a short, calming bedtime lullaby story for a 6-year-old girl."}]}
    ]
  }' | jq -r '.choices[0].message.audio.data' | base64 -d > lullaby.wav

echo "wrote lullaby.wav ($(stat -f%z lullaby.wav 2>/dev/null || stat -c%s lullaby.wav) bytes)"
# Cleanup:  docker stop "${NAME}" && docker rm "${NAME}"
