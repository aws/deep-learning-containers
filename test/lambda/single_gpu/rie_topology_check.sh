#!/usr/bin/env bash
# Verify RIC concurrency-mode process/thread topology LOCALLY via the Runtime
# Interface Emulator (RIE) — NOT real Lambda managed instances. (For the real
# managed-GPU equivalent see test/lambda/platform/.)
#
# It mirrors the real-Lambda test's pattern for every preview image by layering
# test/lambda/platform/test_handler.py over the image (mounted + CMD override,
# so it works whether or not the image ships a baked serving handler):
#
#   * base/cupy/pytorch-cuda (--engine none): fire N concurrent `get_pid` probes and
#     count worker processes/threads per mode.
#   * vllm/sglang (--engine vllm|sglang): fire N concurrent REAL inferences
#     (`infer_probe`), assert each returns a completion, count processes/threads, and
#     assert exactly ONE shared engine process on the GPU (workers proxy to it).
#
# Expected worker topology (vcpus = nproc inside the container):
#     thread  -> 1 process,      N threads
#     process -> N processes,    1 thread each
#     hybrid  -> vcpus processes, ~N/vcpus threads each
#
# Usage: rie_topology_check.sh <image> [N] [modes] [engine]
#   N       concurrent invocations (default 8; keep > vcpus so hybrid != process)
#   modes   subset of {thread,process,hybrid} (default all three)
#   engine  none|vllm|sglang (default none)
set -uo pipefail

IMAGE="${1:?image uri required}"
N="${2:-8}"
MODES="${3:-thread process hybrid}"; MODES="${MODES//,/ }"
ENGINE="${4:-none}"
MODEL="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT=9000
INVOKE_URL="http://localhost:${PORT}/2015-03-31/functions/function/invocations"
HANDLER_SRC="$(cd "$(dirname "$0")/../platform" && pwd)/test_handler.py"
RC=0

MODEL_MOUNT=""
if [ "${ENGINE}" = "none" ]; then
  PAYLOAD='{"action":"get_pid","sleep":3}'
  READY_PAYLOAD='{"action":"get_pid","sleep":0}'
  READY_TIMEOUT=60
else
  PAYLOAD="{\"action\":\"infer_probe\",\"payload\":{\"prompt\":\"The capital of France is\",\"max_tokens\":16}}"
  READY_PAYLOAD="${PAYLOAD}"
  READY_TIMEOUT=600   # engine cold start: flashinfer JIT + engine warmup
  # RIE runs locally, so mount the model from S3 (no HuggingFace dependency): download
  # + extract the tarball once and bind-mount it; the engine serves it as a local path.
  MODEL_S3_URI="${MODEL_S3_URI:-s3://dlc-cicd-models/llm-models/qwen3-0.6b.tar.gz}"
  MODEL_DIR="$(mktemp -d)"
  echo "fetching model ${MODEL_S3_URI} ..."
  aws s3 cp "${MODEL_S3_URI}" "${MODEL_DIR}/model.tar.gz" >/dev/null
  tar -xzf "${MODEL_DIR}/model.tar.gz" -C "${MODEL_DIR}" && rm -f "${MODEL_DIR}/model.tar.gz"
  MODEL_HOST="$(dirname "$(find "${MODEL_DIR}" -name config.json | head -1)")"
  if [ -z "${MODEL_HOST}" ] || [ "${MODEL_HOST}" = "." ]; then
    echo "FAIL: could not locate model config.json under extracted tarball"; rm -rf "${MODEL_DIR}"; exit 1
  fi
  MODEL="/opt/model"
  MODEL_MOUNT="-v ${MODEL_HOST}:/opt/model:ro"
fi
trap '[ -n "${MODEL_DIR:-}" ] && rm -rf "${MODEL_DIR}"' EXIT

echo "image=${IMAGE} N=${N} modes=[${MODES}] engine=${ENGINE} model=${MODEL}"

for MODE in ${MODES}; do
  echo "########## MODE=${MODE} ##########"
  C="rie-topo-${MODE}"
  docker rm -f "$C" >/dev/null 2>&1
  docker run -d --name "$C" --gpus all -p ${PORT}:8080 \
    -e AWS_LAMBDA_CONCURRENCY_MODE="${MODE}" -e AWS_LAMBDA_MAX_CONCURRENCY="${N}" \
    -e MODEL_ID="${MODEL}" -e HF_HOME=/tmp/hf \
    -e VLLM_GPU_MEM_UTIL=0.4 -e VLLM_MAX_MODEL_LEN=2048 \
    -e SGLANG_MEM_FRACTION=0.4 -e SGLANG_MAX_TOTAL_TOKENS=2048 \
    -v "${HANDLER_SRC}:/var/task/test_handler.py:ro" ${MODEL_MOUNT} \
    "${IMAGE}" test_handler.handler >/dev/null

  # Wait until the handler actually responds successfully. NOTE: the RIE returns
  # HTTP 200 even when the handler raises (body carries errorMessage), so `curl -f`
  # is not enough — require a real pid/ok in the JSON body (engine: server warmed).
  ready=0
  for _ in $(seq 1 $((READY_TIMEOUT / 5))); do
    RESP=$(curl -s -m 30 "${INVOKE_URL}" -d "${READY_PAYLOAD}" 2>/dev/null)
    if echo "${RESP}" | jq -e '(.pid != null) or (.ok == true)' >/dev/null 2>&1; then ready=1; break; fi
    sleep 5
  done
  if [ "${ready}" != "1" ]; then
    echo "FAIL mode=${MODE}: handler not ready within ${READY_TIMEOUT}s"; docker logs "$C" 2>&1 | tail -20; RC=1
    docker rm -f "$C" >/dev/null 2>&1; continue
  fi

  # Container is up now — read vcpus (RIC uses it to size hybrid). Fail loudly if unknown.
  VCPUS=$(docker exec "$C" nproc 2>/dev/null)
  if ! [[ "${VCPUS}" =~ ^[0-9]+$ ]]; then
    echo "FAIL mode=${MODE}: could not read container nproc"; RC=1; docker rm -f "$C" >/dev/null 2>&1; continue
  fi
  case "${MODE}" in
    thread) EXP_PROCS=1 ;;
    process) EXP_PROCS="${N}" ;;
    hybrid) EXP_PROCS="${VCPUS}" ;;
  esac

  # Boot every worker before measuring. In process/hybrid mode the RIC brings its
  # worker processes up lazily, and on a small host not all are ready at once — RIE
  # then answers a concurrent burst with "no idle runtimes". Fire cheap concurrent
  # probes and retry until all N workers respond, so the measured burst below reflects
  # the real topology instead of a half-warmed one.
  warm=0
  for attempt in $(seq 1 20); do
    wtmp=$(mktemp -d)
    for i in $(seq 1 "${N}"); do
      ( curl -s -m 30 "${INVOKE_URL}" -d '{"action":"get_pid","sleep":1}' > "${wtmp}/${i}.json" ) &
    done
    wait
    warm=$(grep -l '"pid"' "${wtmp}"/*.json 2>/dev/null | wc -l | tr -d ' ')
    rm -rf "${wtmp}"
    [ "${warm}" -ge "${N}" ] && break
    sleep 3
  done
  echo "warmup: ${warm}/${N} workers ready after ${attempt} attempt(s)"

  # Fire N concurrent invokes; each reports the worker's pid/tid (and ok for engines).
  tmp=$(mktemp -d)
  for i in $(seq 1 "${N}"); do
    ( curl -s -m 600 "${INVOKE_URL}" -d "${PAYLOAD}" > "${tmp}/${i}.json" ) &
  done
  wait

  # Parse per-file so one truncated/timed-out response can't corrupt the rest.
  PAIRS=""; OK_COUNT=0
  for f in "${tmp}"/*.json; do
    p=$(jq -r 'select(.pid!=null) | "\(.pid) \(.tid)"' "$f" 2>/dev/null)
    [ -n "$p" ] && PAIRS+="${p}"$'\n'
    [ "${ENGINE}" != "none" ] && jq -e '.ok==true' "$f" >/dev/null 2>&1 && OK_COUNT=$((OK_COUNT + 1))
  done
  PAIRS=$(printf '%s' "${PAIRS}" | sort -u)                                  # dedupe -> distinct (pid,tid)
  HANDLERS=$(printf '%s\n' "${PAIRS}" | grep -c '[0-9]')                      # distinct concurrent handlers
  PROCS=$(printf '%s\n' "${PAIRS}" | awk 'NF{print $1}' | sort -u | grep -c '[0-9]')
  if [ "${ENGINE}" != "none" ]; then
    GPUPROCS=$(docker exec "$C" nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]' || echo "?")
  fi
  rm -rf "${tmp}"

  echo "observed: procs=${PROCS} (expected ${EXP_PROCS}), handlers=${HANDLERS}/${N}, vcpus=${VCPUS}${GPUPROCS:+, gpu_procs=${GPUPROCS}}"

  PASS=1
  [ "${PROCS}" = "${EXP_PROCS}" ] || { echo "  procs ${PROCS} != expected ${EXP_PROCS}"; PASS=0; }
  [ "${HANDLERS}" = "${N}" ] || { echo "  only ${HANDLERS}/${N} handlers responded"; PASS=0; }
  if [ "${ENGINE}" != "none" ]; then
    [ "${OK_COUNT}" = "${N}" ] || { echo "  only ${OK_COUNT}/${N} inferences returned a completion"; PASS=0; }
    [ "${GPUPROCS}" = "1" ] || { echo "  gpu_procs ${GPUPROCS} != 1 (shared-model invariant)"; PASS=0; }
  fi
  if [ "${PASS}" = "1" ]; then echo "PASS mode=${MODE}"; else echo "FAIL mode=${MODE}"; docker logs "$C" 2>&1 | grep -iE 'error|out of memory|Uvicorn|server' | tail -15; RC=1; fi
  docker rm -f "$C" >/dev/null 2>&1
done
exit "${RC}"
