#!/usr/bin/env bash
# End-to-end sync video-generation example: start server, submit, get MP4 back.
# /v1/videos/sync (new in 0.20.0) blocks until the video is ready and returns
# raw MP4 bytes — no job-ID polling needed, unlike async /v1/videos.
set -euo pipefail

IMAGE="${IMAGE:-763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm-omni:omni-cuda-v1}"
MODEL="${MODEL:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
NAME="${NAME:-omni-video-sync}"

docker run -d --name "${NAME}" --gpus all --shm-size=8g -p 8080:8000 \
  -v "${HOME}/hf-cache:/root/.cache/huggingface" \
  "${IMAGE}" --model "${MODEL}" --tensor-parallel-size 2

until curl -sf http://localhost:8080/health >/dev/null; do sleep 5; done

# /v1/videos/sync requires multipart/form-data and blocks until the MP4 is ready.
curl -sf -X POST http://localhost:8080/v1/videos/sync \
  -F "prompt=a dog running on a beach at sunset" \
  -F "num_frames=17" -F "num_inference_steps=30" \
  -F "size=480x320" -F "seed=42" \
  --output video.mp4

echo "wrote video.mp4 ($(stat -f%z video.mp4 2>/dev/null || stat -c%s video.mp4) bytes)"
# Cleanup:  docker stop "${NAME}" && docker rm "${NAME}"
