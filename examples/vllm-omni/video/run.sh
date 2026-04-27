#!/usr/bin/env bash
# End-to-end video-generation example: start server, submit job, poll, download.
# /v1/videos is async — it returns a job ID; the MP4 is produced in the background.
set -euo pipefail

IMAGE="${IMAGE:-763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm-omni:omni-cuda-v1}"
MODEL="${MODEL:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
NAME="${NAME:-omni-video}"

docker run -d --name "${NAME}" --gpus all --shm-size=8g -p 8080:8000 \
  -v "${HOME}/hf-cache:/root/.cache/huggingface" \
  "${IMAGE}" --model "${MODEL}"

until curl -sf http://localhost:8080/health >/dev/null; do sleep 5; done

# /v1/videos requires multipart/form-data.
JOB_ID=$(curl -sf -X POST http://localhost:8080/v1/videos \
  -F "prompt=a dog running on a beach at sunset" \
  -F "num_frames=17" -F "num_inference_steps=4" \
  -F "size=480x320" -F "seed=42" \
  | python3 -c "import json,sys;print(json.load(sys.stdin)['id'])")

echo "submitted job ${JOB_ID}"

# Poll until succeeded (5s interval, 10 min timeout).
for _ in $(seq 1 120); do
  STATUS=$(curl -sf "http://localhost:8080/v1/videos/${JOB_ID}" \
    | python3 -c "import json,sys;print(json.load(sys.stdin)['status'])")
  [ "${STATUS}" = "succeeded" ] && break
  [ "${STATUS}" = "failed" ] && { echo "job failed"; exit 1; }
  sleep 5
done

curl -sf "http://localhost:8080/v1/videos/${JOB_ID}/content" --output video.mp4
echo "wrote video.mp4 ($(stat -f%z video.mp4 2>/dev/null || stat -c%s video.mp4) bytes)"
# Cleanup:  docker stop "${NAME}" && docker rm "${NAME}"
