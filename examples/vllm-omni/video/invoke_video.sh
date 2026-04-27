#!/bin/bash
# Video generation via /v1/videos endpoint (async — returns a job ID)
# The /v1/videos API requires multipart/form-data.
JOB=$(curl -sf -X POST http://localhost:8080/v1/videos \
  -F "prompt=a dog running on a beach" \
  -F "num_frames=17" \
  -F "num_inference_steps=4" \
  -F "size=480x320" \
  -F "seed=42")

JOB_ID=$(echo "$JOB" | python3 -c "import json,sys;print(json.load(sys.stdin)['id'])")
echo "Job: $JOB_ID"

# Poll until complete, then download
while true; do
  STATUS=$(curl -sf "http://localhost:8080/v1/videos/$JOB_ID" | python3 -c "import json,sys;print(json.load(sys.stdin)['status'])")
  [ "$STATUS" = "succeeded" ] && break
  [ "$STATUS" = "failed" ] && { echo "Job failed"; exit 1; }
  sleep 5
done

curl -sf "http://localhost:8080/v1/videos/$JOB_ID/content" --output video.mp4
