#!/usr/bin/env bash
# Simple container check: verify a vLLM/sglang lambda serving image can stream model
# weights DIRECTLY FROM S3 via runai-model-streamer and serve a completion — run
# LOCALLY on the GPU test runner through the Runtime Interface Emulator (RIE). No
# Lambda, no managed instances; this just proves the S3 load mechanism works.
#
# Usage: runai_s3_check.sh <image> <engine> [model_s3_uri]
#   image         vLLM/sglang lambda serving image uri
#   engine        vllm | sglang
#   model_s3_uri  s3://bucket/prefix/ of RAW safetensors (config.json + *.safetensors);
#                 defaults to $MODEL_S3_URI. NOT a tarball — the engine reads it live.
#
# Requires on the runner: a GPU, docker, jq, and AWS credentials in the environment
# (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN + AWS_REGION) with
# s3:GetObject on the model bucket — passed into the container so runai can read S3.
set -uo pipefail

IMAGE="${1:?image uri required}"
ENGINE="${2:?engine (vllm|sglang) required}"
MODEL_S3="${3:-${MODEL_S3_URI:?raw-safetensors s3:// uri required (arg 3 or MODEL_S3_URI)}}"

PORT=9000
INVOKE_URL="http://localhost:${PORT}/2015-03-31/functions/function/invocations"
NAME="runai-s3-check-${ENGINE}"
SERVED="runai-s3-model"
READY_TIMEOUT=600
PAYLOAD="{\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"model\":\"${SERVED}\"}"

case "${ENGINE}" in
  vllm)   ENGINE_ENV=(-e VLLM_LOAD_FORMAT=runai_streamer -e VLLM_GPU_MEM_UTIL=0.4 -e VLLM_MAX_MODEL_LEN=2048) ;;
  sglang) ENGINE_ENV=(-e SGLANG_LOAD_FORMAT=runai_streamer -e SGLANG_MEM_FRACTION=0.4 -e SGLANG_MAX_TOTAL_TOKENS=2048) ;;
  *) echo "FAIL: unknown engine '${ENGINE}' (expected vllm|sglang)"; exit 2 ;;
esac

docker rm -f "${NAME}" >/dev/null 2>&1
trap 'docker rm -f "${NAME}" >/dev/null 2>&1 || true' EXIT

echo "image=${IMAGE} engine=${ENGINE} model=${MODEL_S3}"
docker run -d --name "${NAME}" --gpus all -p ${PORT}:8080 \
  -e MODEL_ID="${MODEL_S3}" \
  -e SERVED_MODEL_NAME="${SERVED}" \
  "${ENGINE_ENV[@]}" \
  -e AWS_REGION="${AWS_REGION:-us-east-2}" \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_SESSION_TOKEN \
  "${IMAGE}" handler.handler >/dev/null

# Cold start streams the weights from S3 then warms the engine. RIE returns HTTP 200
# even when the handler raises (body carries errorMessage), so require a real
# completion (.choices[0].text) in the body — that only appears once the model has
# loaded from S3 and served the request.
BODY=""
for _ in $(seq 1 $((READY_TIMEOUT / 5))); do
  if ! docker ps -q -f name="${NAME}" | grep -q .; then
    echo "FAIL: container exited during startup"; docker logs --tail 60 "${NAME}"; exit 1
  fi
  BODY="$(curl -s -m 60 "${INVOKE_URL}" -d "${PAYLOAD}" 2>/dev/null)"
  if echo "${BODY}" | jq -e '((.choices[0].text // "") | length) > 0' >/dev/null 2>&1; then
    echo "PASS: ${ENGINE} streamed weights from S3 and served a completion:"
    echo "      '$(echo "${BODY}" | jq -r '.choices[0].text')'"
    exit 0
  fi
  sleep 5
done

echo "FAIL: no completion within ${READY_TIMEOUT}s (S3 stream / serve failed)"
echo "last response: ${BODY:0:400}"
docker logs --tail 80 "${NAME}" 2>&1
exit 1
