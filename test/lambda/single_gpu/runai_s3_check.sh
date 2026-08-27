#!/usr/bin/env bash
# Verify runai-model-streamer S3 streaming LOCALLY via the Runtime Interface
# Emulator (RIE) for the vLLM / sglang serving images. (For the real managed-GPU
# equivalent see test/lambda/platform/run_runai_s3.py.)
#
# Runs the image's DEFAULT serving handler with MODEL_ID pointed at an s3:// prefix
# and <ENGINE>_LOAD_FORMAT=runai_streamer, so the engine streams safetensors straight
# from S3 into GPU memory (no HuggingFace download, no /tmp staging), then fires one
# inference and asserts a real completion comes back.
#
# The model fixture is raw safetensors at ${RAW_S3_PREFIX}. If that prefix is missing
# it is staged ONCE (idempotently) by extracting ${MODEL_TARBALL_S3} and uploading the
# raw files — subsequent runs stream read-only.
#
# Usage: runai_s3_check.sh <image> <engine:vllm|sglang>
set -uo pipefail

IMAGE="${1:?image uri required}"
ENGINE="${2:?engine (vllm|sglang) required}"
case "${ENGINE}" in vllm|sglang) ;; *) echo "engine must be vllm|sglang"; exit 2 ;; esac

MODELS_BUCKET="${MODELS_BUCKET:-dlc-cicd-models}"
MODEL_TARBALL_S3="${MODEL_TARBALL_S3:-s3://${MODELS_BUCKET}/llm-models/qwen3-0.6b.tar.gz}"
RAW_S3_PREFIX="${RAW_S3_PREFIX:-s3://${MODELS_BUCKET}/llm-models-safetensors/qwen3-0.6b}"
SERVED_NAME="${SERVED_MODEL_NAME:-qwen3-0.6b}"
PORT=8080                  # RIE listens on 8080; under --network host it binds the host directly
INVOKE_URL="http://localhost:${PORT}/2015-03-31/functions/function/invocations"
READY_TIMEOUT=600          # engine cold start: S3 stream + flashinfer JIT + warmup
PAYLOAD='{"prompt":"The capital of France is","max_tokens":16}'
RC=0

if ! aws s3 ls "${RAW_S3_PREFIX}/config.json" >/dev/null 2>&1; then
  echo "staging raw safetensors fixture at ${RAW_S3_PREFIX} (one-time) from ${MODEL_TARBALL_S3} ..."
  STAGE_DIR="$(mktemp -d)"
  aws s3 cp "${MODEL_TARBALL_S3}" "${STAGE_DIR}/model.tar.gz" >/dev/null || { echo "FAIL: cannot fetch ${MODEL_TARBALL_S3}"; rm -rf "${STAGE_DIR}"; exit 1; }
  tar -xzf "${STAGE_DIR}/model.tar.gz" -C "${STAGE_DIR}" && rm -f "${STAGE_DIR}/model.tar.gz"
  SRC_DIR="$(dirname "$(find "${STAGE_DIR}" -name config.json | head -1)")"
  if [ -z "${SRC_DIR}" ] || [ "${SRC_DIR}" = "." ]; then echo "FAIL: no config.json in tarball"; rm -rf "${STAGE_DIR}"; exit 1; fi
  aws s3 sync "${SRC_DIR}" "${RAW_S3_PREFIX}/" --exclude ".cache/*" >/dev/null || { echo "FAIL: cannot upload fixture to ${RAW_S3_PREFIX}"; rm -rf "${STAGE_DIR}"; exit 1; }
  rm -rf "${STAGE_DIR}"
fi
echo "image=${IMAGE} engine=${ENGINE} model=${RAW_S3_PREFIX} served_name=${SERVED_NAME}"

C="runai-s3-${ENGINE}"
docker rm -f "$C" >/dev/null 2>&1
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-2}}"

# runai-model-streamer reads S3 from *inside* the container, which by default has
# no route to the runner's credential endpoints. Share the host network so the
# ECS container-creds endpoint (169.254.170.2) and EC2 IMDS (169.254.169.254) are
# reachable, then forward the provider-chain env vars so boto3 in the container
# resolves the same CI role. Vars are passed by NAME (-e VAR, no value) so nothing
# sensitive lands in argv / CI logs; the bearer token is masked defensively.
[ -n "${AWS_CONTAINER_CREDENTIALS_TOKEN:-}" ] && echo "::add-mask::${AWS_CONTAINER_CREDENTIALS_TOKEN}"
CREDS_ENV=""
for v in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN \
         AWS_CONTAINER_CREDENTIALS_RELATIVE_URI AWS_CONTAINER_CREDENTIALS_FULL_URI \
         AWS_CONTAINER_CREDENTIALS_TOKEN; do
  [ -n "${!v:-}" ] && CREDS_ENV+=" -e ${v}"
done

# --network host => the RIE binds ${PORT} on the host directly (no -p mapping).
docker run -d --name "$C" --gpus all --network host \
  -e MODEL_ID="${RAW_S3_PREFIX}" \
  -e "${ENGINE^^}_LOAD_FORMAT=runai_streamer" \
  -e SERVED_MODEL_NAME="${SERVED_NAME}" \
  -e VLLM_GPU_MEM_UTIL=0.4 -e VLLM_MAX_MODEL_LEN=2048 \
  -e SGLANG_MEM_FRACTION=0.4 -e SGLANG_MAX_TOTAL_TOKENS=2048 \
  -e AWS_REGION="${REGION}" ${CREDS_ENV} \
  "${IMAGE}" >/dev/null

# RIE returns HTTP 200 even when the handler raises (error in the body), so gate
# readiness on a non-empty completion rather than the HTTP status.
COMPLETION=""
for _ in $(seq 1 $((READY_TIMEOUT / 5))); do
  RESP=$(curl -s -m 60 "${INVOKE_URL}" -d "${PAYLOAD}" 2>/dev/null)
  TXT=$(printf '%s' "${RESP}" | jq -r '.choices[0].text // empty' 2>/dev/null)
  if [ -n "${TXT}" ]; then COMPLETION="${TXT}"; break; fi
  sleep 5
done

if [ -n "${COMPLETION}" ]; then
  echo "PASS ${ENGINE}: streamed ${RAW_S3_PREFIX} via runai_streamer -> completion: ${COMPLETION}"
else
  echo "FAIL ${ENGINE}: no completion within ${READY_TIMEOUT}s (runai S3 stream / serving failed)"
  docker logs "$C" 2>&1 | grep -iE 'runai|s3|error|out of memory|Uvicorn|server|load' | tail -25
  RC=1
fi
docker rm -f "$C" >/dev/null 2>&1
exit "${RC}"
