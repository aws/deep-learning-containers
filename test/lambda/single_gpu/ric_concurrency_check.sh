#!/usr/bin/env bash
# Verify the RIC invoke lifecycle LOCALLY via the Runtime Interface Emulator (RIE).
#
# Unlike the pytest suites, this does NOT override the image entrypoint, so the real
# chain runs: entrypoint shim -> aws-lambda-rie -> python -m awslambdaric -> handler.
#
# Two shapes, matching awslambdaric 4.1.0 (no AWS_LAMBDA_CONCURRENCY_MODE):
#   multi     AWS_LAMBDA_MAX_CONCURRENCY=N -> exactly N forked workers, pre-fork hooks run
#   ondemand  unset                        -> one worker, no pre-fork hooks
#
# Usage: ric_concurrency_check.sh <image> [N] [engine:none|vllm|sglang]
set -uo pipefail

IMAGE="${1:?image uri required}"
N="${2:-4}"
ENGINE="${3:-none}"
case "${ENGINE}" in none | vllm | sglang) ;; *) echo "engine must be none|vllm|sglang"; exit 2 ;; esac

PORT="${PORT:-9000}"
INVOKE_URL="http://localhost:${PORT}/2015-03-31/functions/function/invocations"
HANDLER_SRC="$(cd "$(dirname "$0")" && pwd)/ric_probe_handler.py"
CONTAINER=ric-check
RC=0
SHAPE_RC=0

# Overridable so the check can run on a CPU box during development.
if [ -z "${DOCKER_GPU_FLAG+x}" ]; then GPU_FLAG="--gpus all"; else GPU_FLAG="${DOCKER_GPU_FLAG}"; fi

REQUIRED_LIBS="awslambdaric,boto3"
[ "${ENGINE}" != "none" ] && REQUIRED_LIBS="${REQUIRED_LIBS},${ENGINE}"

MODEL_MOUNT=""
MODEL_ID="none"
READY_TIMEOUT=90
if [ "${ENGINE}" != "none" ]; then
  READY_TIMEOUT=900 # engine cold start: load + warmup
  # Mount the model so the engine needs no HuggingFace access from the runner.
  MODEL_S3_URI="${MODEL_S3_URI:-s3://dlc-cicd-models/llm-models/qwen3-0.6b.tar.gz}"
  MODEL_DIR="$(mktemp -d)"
  echo "fetching model ${MODEL_S3_URI} ..."
  aws s3 cp "${MODEL_S3_URI}" "${MODEL_DIR}/model.tar.gz" >/dev/null || {
    echo "FAIL: cannot fetch ${MODEL_S3_URI}"
    rm -rf "${MODEL_DIR}"
    exit 1
  }
  tar -xzf "${MODEL_DIR}/model.tar.gz" -C "${MODEL_DIR}" && rm -f "${MODEL_DIR}/model.tar.gz"
  MODEL_HOST="$(dirname "$(find "${MODEL_DIR}" -name config.json | head -1)")"
  if [ -z "${MODEL_HOST}" ] || [ "${MODEL_HOST}" = "." ]; then
    echo "FAIL: no config.json in ${MODEL_S3_URI}"
    rm -rf "${MODEL_DIR}"
    exit 1
  fi
  MODEL_ID="/opt/model"
  MODEL_MOUNT="-v ${MODEL_HOST}:/opt/model:ro"
fi
trap 'docker rm -f "${CONTAINER}" >/dev/null 2>&1; [ -n "${MODEL_DIR:-}" ] && rm -rf "${MODEL_DIR}"' EXIT

echo "image=${IMAGE} N=${N} engine=${ENGINE} libs=${REQUIRED_LIBS}"

invoke() { curl -s -m "${2:-600}" "${INVOKE_URL}" -d "$1" 2>/dev/null; }

# 1 if the JSON satisfies the jq filter, else 0.
jq_ok() { echo "$1" | jq -e "$2" >/dev/null 2>&1 && echo 1 || echo 0; }

check() { # check <name> <1|0> [detail]
  if [ "$2" = "1" ]; then
    echo "  PASS  $1"
  else
    echo "  FAIL  $1${3:+ — $3}"
    RC=1
    SHAPE_RC=1
  fi
}

start_container() { # start_container <multi|ondemand>
  docker rm -f "${CONTAINER}" >/dev/null 2>&1
  local conc=()
  [ "$1" = "multi" ] && conc=(-e "AWS_LAMBDA_MAX_CONCURRENCY=${N}")
  # shellcheck disable=SC2086
  docker run -d --name "${CONTAINER}" ${GPU_FLAG} -p "${PORT}:8080" \
    "${conc[@]}" \
    -e MODEL_ID="${MODEL_ID}" -e HF_HOME=/tmp/hf \
    -e VLLM_GPU_MEM_UTIL=0.4 -e VLLM_MAX_MODEL_LEN=2048 \
    -e SGLANG_MEM_FRACTION=0.4 -e SGLANG_MAX_TOTAL_TOKENS=2048 \
    -v "${HANDLER_SRC}:/var/task/ric_probe_handler.py:ro" ${MODEL_MOUNT} \
    "${IMAGE}" ric_probe_handler.handler >/dev/null
}

wait_ready() {
  # The RIE answers 200 even when the handler raises, so require a real field.
  local deadline=$((SECONDS + READY_TIMEOUT))
  while [ "${SECONDS}" -lt "${deadline}" ]; do
    if [ "$(jq_ok "$(invoke '{"action":"get_pid","sleep":0}' 60)" '.pid != null')" = "1" ]; then
      return 0
    fi
    sleep 5
  done
  echo "  handler not ready within ${READY_TIMEOUT}s"
  docker logs "${CONTAINER}" 2>&1 | tail -20
  return 1
}

# Fires n concurrent invokes; sets BURST_PAIRS ("pid tid" lines, deduped) and BURST_OK.
burst() { # burst <n> <payload>
  local n="$1" payload="$2" tmp i f p
  tmp="$(mktemp -d)"
  for i in $(seq 1 "${n}"); do
    (invoke "${payload}" >"${tmp}/${i}.json") &
  done
  wait
  BURST_PAIRS=""
  BURST_OK=0
  for f in "${tmp}"/*.json; do
    p="$(jq -r 'select(.pid!=null) | "\(.pid) \(.tid)"' "${f}" 2>/dev/null)"
    [ -n "${p}" ] && BURST_PAIRS+="${p}"$'\n'
    jq -e '.ok==true' "${f}" >/dev/null 2>&1 && BURST_OK=$((BURST_OK + 1))
  done
  BURST_PAIRS="$(printf '%s' "${BURST_PAIRS}" | sort -u)"
  rm -rf "${tmp}"
}

count_procs() { printf '%s\n' "${BURST_PAIRS}" | awk 'NF{print $1}' | sort -u | grep -c '[0-9]'; }
count_handlers() { printf '%s\n' "${BURST_PAIRS}" | grep -c '[0-9]'; }

gpu_procs() {
  docker exec "${CONTAINER}" nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null |
    grep -c '[0-9]'
}

dump_logs_on_failure() {
  [ "${SHAPE_RC}" = "0" ] ||
    docker logs "${CONTAINER}" 2>&1 | grep -iE 'error|traceback|out of memory' | tail -15
}

# --------------------------------------------------------------------------- multi
echo "########## shape=multi (AWS_LAMBDA_MAX_CONCURRENCY=${N}) ##########"
SHAPE_RC=0
start_container multi
if ! wait_ready; then
  RC=1
  SHAPE_RC=1
else
  # The invoke round-trips through the RIC unchanged.
  RESP="$(invoke '{"action":"echo","msg":"hello"}' 60)"
  check "echo invoke round-trip" "$(jq_ok "${RESP}" '.msg=="hello"')" "got ${RESP}"

  # DLC libraries import from handler code reached through the RIC.
  LIBS_JSON="$(printf '%s' "${REQUIRED_LIBS}" | jq -R 'split(",")')"
  RESP="$(invoke "{\"action\":\"import_check\",\"libs\":${LIBS_JSON}}" 180)"
  check "imports from handler (${REQUIRED_LIBS})" \
    "$(jq_ok "${RESP}" 'to_entries | all(.value==true)')" "got ${RESP}"

  # Workers start lazily; the RIE rejects a burst with "no idle runtimes" until all N are up.
  for attempt in $(seq 1 20); do
    burst "${N}" '{"action":"get_pid","sleep":1}'
    [ "$(count_procs)" -ge "${N}" ] && break
    sleep 3
  done
  echo "warmup: $(count_procs)/${N} workers after ${attempt} attempt(s)"

  # N concurrent invokes land on N distinct forked workers.
  if [ "${ENGINE}" = "none" ]; then
    burst "${N}" '{"action":"get_pid","sleep":3}'
  else
    burst "${N}" '{"action":"infer_probe","payload":{"prompt":"The capital of France is","max_tokens":16}}'
  fi
  HANDLERS="$(count_handlers)"
  PROCS="$(count_procs)"
  check "${N} concurrent invokes served" \
    "$([ "${HANDLERS}" = "${N}" ] && echo 1 || echo 0)" "${HANDLERS}/${N} responded"
  check "exactly ${N} worker processes" \
    "$([ "${PROCS}" = "${N}" ] && echo 1 || echo 0)" "observed ${PROCS}"

  # The pre-fork hook ran once, in the parent, before the workers forked.
  RESP="$(invoke '{"action":"check_hook"}' 60)"
  check "register_pre_fork ran in the parent" \
    "$(jq_ok "${RESP}" '.hook_ran_in_parent==true')" "got ${RESP}"

  if [ "${ENGINE}" != "none" ]; then
    check "${N} real inferences returned a completion" \
      "$([ "${BURST_OK}" = "${N}" ] && echo 1 || echo 0)" "${BURST_OK}/${N} ok"
    GPU="$(gpu_procs)"
    check "one shared GPU process" "$([ "${GPU}" = "1" ] && echo 1 || echo 0)" "gpu_procs=${GPU}"
  fi
fi
dump_logs_on_failure
docker rm -f "${CONTAINER}" >/dev/null 2>&1

# ------------------------------------------------------------------------ ondemand
echo "########## shape=ondemand (no AWS_LAMBDA_MAX_CONCURRENCY) ##########"
SHAPE_RC=0
start_container ondemand
if ! wait_ready; then
  RC=1
  SHAPE_RC=1
else
  RESP="$(invoke '{"action":"echo","msg":"hello"}' 60)"
  check "echo invoke round-trip" "$(jq_ok "${RESP}" '.msg=="hello"')" "got ${RESP}"

  # One worker, so two sequential invokes report the same PID. No burst: the RIE has a
  # single runtime here and would reject concurrent invokes.
  PID1="$(invoke '{"action":"get_pid","sleep":0}' 60 | jq -r '.pid // empty')"
  PID2="$(invoke '{"action":"get_pid","sleep":0}' 60 | jq -r '.pid // empty')"
  check "single worker process" \
    "$([ -n "${PID1}" ] && [ "${PID1}" = "${PID2}" ] && echo 1 || echo 0)" \
    "pids ${PID1:-?} ${PID2:-?}"

  # No fork, so no pre-fork hook: the engine handlers start their server at module level.
  RESP="$(invoke '{"action":"check_hook"}' 60)"
  check "no pre-fork hook without MAX_CONCURRENCY" \
    "$(jq_ok "${RESP}" '.hook_executed==false')" "got ${RESP}"

  if [ "${ENGINE}" != "none" ]; then
    RESP="$(invoke '{"action":"infer_probe","payload":{"prompt":"The capital of France is","max_tokens":16}}')"
    check "real inference returned a completion" "$(jq_ok "${RESP}" '.ok==true')" "got ${RESP}"
    GPU="$(gpu_procs)"
    check "one shared GPU process" "$([ "${GPU}" = "1" ] && echo 1 || echo 0)" "gpu_procs=${GPU}"
  fi
fi
dump_logs_on_failure
docker rm -f "${CONTAINER}" >/dev/null 2>&1

if [ "${RC}" = "0" ]; then echo "ALL CHECKS PASSED"; else echo "CHECKS FAILED"; fi
exit "${RC}"
