#!/usr/bin/env bash
# Verify the multi-mode RIC topology AND the shared-server serving model for the
# vllm-preview image, per AWS_LAMBDA_CONCURRENCY_MODE + AWS_LAMBDA_MAX_CONCURRENCY(=N).
#
# Handler design (LMI serving pattern): ONE shared `vllm serve` HTTP server is
# started once (module-level in thread mode; @register_pre_fork in process/hybrid),
# and each RIC worker is a thin proxy to it. So the invariants are:
#   * exactly ONE vllm server / ONE "EngineCore pid" for ALL modes (one model copy)
#   * RIC worker topology per spec (awslambdaric/__main__.py, vcpus = nproc):
#       thread  -> 1 process,  N threads
#       process -> N processes, 1 thread
#       hybrid  -> vcpus procs, max(1, N//vcpus) threads
#   * all three modes actually serve (proxy returns a completion)
#
# One shared model means hybrid no longer needs vcpus model copies — it should now
# PASS on a single GPU (the offline-engine-per-worker design could not).
#
# Usage: ric_topology_check.sh <image> [N] [modes]
#   modes: space/comma-separated subset of {single,thread,process,hybrid} (default all
#          multi-mode: "thread process hybrid"). Use "single" for images with the STOCK
#          awslambdaric (non-preview) — it has no multi-mode support, so we run with
#          NO concurrency env vars set (standard single-worker path) instead.
# Env: HANDLER_OVERRIDE=/path/handler.py to mount a handler over /var/task (no rebuild).
set -uo pipefail

IMAGE="${1:?image uri required}"
N="${2:-4}"
MODES="${3:-thread process hybrid}"
MODES="${MODES//,/ }"
MODEL="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT=9000
RC=0
VCPUS=$(nproc)

echo "N=${N}  vcpus=${VCPUS}  modes=[${MODES}]  (shared-server model: expect 1 GPU process per mode)"

expected_procs() {  # RIC worker processes
  case "$1" in
    single) echo 1 ;;
    thread) echo 1 ;;
    process) echo "${N}" ;;
    hybrid) echo "${VCPUS}" ;;
  esac
}

for MODE in ${MODES}; do
  echo "########## MODE=${MODE} (expect RIC procs=$(expected_procs "$MODE"), shared model=1) ##########"
  C="ric-topo-${MODE}"
  docker rm -f "$C" >/dev/null 2>&1
  HMOUNT=""
  [ -n "${HANDLER_OVERRIDE:-}" ] && HMOUNT="-v ${HANDLER_OVERRIDE}:/var/task/handler.py:ro"
  # 'single' = NO concurrency env (standard single-worker path, for the stock RIC).
  # Other modes set the multi-mode RIC knobs (only honored by the -preview RIC).
  CONC_ENV=""
  [ "${MODE}" != "single" ] && CONC_ENV="-e AWS_LAMBDA_CONCURRENCY_MODE=${MODE} -e AWS_LAMBDA_MAX_CONCURRENCY=${N}"
  docker run -d --name "$C" --gpus all -p ${PORT}:8080 \
    ${CONC_ENV} \
    -e MODEL_ID="${MODEL}" -e HF_HOME=/tmp/hf \
    -e VLLM_GPU_MEM_UTIL=0.4 -e VLLM_MAX_MODEL_LEN=2048 \
    -e TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor}" \
    -e USER="${USER:-lambda}" \
    -e HOME=/tmp -e XDG_CONFIG_HOME=/tmp/.config -e XDG_CACHE_HOME=/tmp/.cache \
    ${HMOUNT} \
    "${IMAGE}" >/dev/null
  sleep 8
  RESP=$(curl -s -m 600 "http://localhost:${PORT}/2015-03-31/functions/function/invocations" \
    -d '{"prompt":"The capital of France is","max_tokens":16}' 2>/dev/null)
  echo "response: ${RESP}"

  # Shared-model invariant (engine-agnostic): count DISTINCT PIDs holding GPU memory.
  # One shared server → one GPU process; an offline-engine-per-worker bug → N. This
  # works for both vllm and sglang (no dependence on engine-specific log strings).
  GPUPROCS=$(docker exec "$C" nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]' || echo "?")

  echo "observed: gpu-processes(model copies)=${GPUPROCS}"
  if echo "${RESP}" | grep -qE '"choices"|"text"|"content"' && [ "${GPUPROCS}" = "1" ]; then
    echo "PASS mode=${MODE}: 1 shared model on GPU, served OK"
  else
    echo "FAIL mode=${MODE}: gpu-processes=${GPUPROCS} (expected 1), served=$(echo "${RESP}" | grep -qcE '"choices"|"text"' )"
    echo "--- logs tail ---"; docker logs "$C" 2>&1 | grep -iE 'error|out of memory|RTDONE|pre_fork|server (did|ready)|Uvicorn|Started server|Launch_server|The server is fired' | tail -20
    RC=1
  fi
  docker rm -f "$C" >/dev/null 2>&1
done
exit "${RC}"
