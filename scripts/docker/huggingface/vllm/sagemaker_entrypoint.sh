#!/usr/bin/env bash
#
# Entrypoint for the SageMaker AI (server-sagemaker, omni-sagemaker) targets.
#
# Mirrors the AWS base contract — maps SM_VLLM_* env vars to vLLM CLI flags,
# auto-detects the model from /opt/ml/model or HF_MODEL_ID, and launches via
# standard-supervisor for process auto-recovery — and layers HF's performance
# defaults on top. Anything the user sets via SM_VLLM_* always wins.

# 1. CUDA forward-compat (sourced: may export LD_LIBRARY_PATH).
if [[ -f /usr/local/bin/start_cuda_compat.sh ]]; then
  source /usr/local/bin/start_cuda_compat.sh || true
fi

# 2. Telemetry (best-effort; present on the AWS base, absent upstream).
[[ -f /usr/local/bin/bash_telemetry.sh ]] && bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

# 3. HF auto-optimization layer (sets env defaults + HF_LMCACHE_KV_CONFIG).
source /usr/local/bin/hf_optimizations.sh

# LMCache: expose the kv-transfer-config through the SM_VLLM_ contract so the
# mapping below turns it into --kv-transfer-config, unless the user set one.
#
# Pass the JSON verbatim. standard-supervisor >=0.1.16 — pinned by the AWS base
# since the vLLM 0.24.0 image — shlex.join()s argv into supervisord's command=,
# which supervisord shlex.split()s back, so the double quotes survive intact.
# Up to 0.1.15 it space-joined instead and the value had to be single-quote
# wrapped to survive; doing that now reaches vLLM with literal quotes and dies in
# json.loads. PROCESS_AUTO_RECOVERY never affected this: it only sets supervisord's
# autorestart, and the command= round-trip happens on every path.
if [[ -n "${HF_LMCACHE_KV_CONFIG:-}" && -z "${SM_VLLM_KV_TRANSFER_CONFIG:-}" ]]; then
  export SM_VLLM_KV_TRANSFER_CONFIG="${HF_LMCACHE_KV_CONFIG}"
fi

# runai-model-streamer (opt-in): default the load format for object-storage models.
if [[ "${HF_ENABLE_RUNAI_STREAMER:-0}" == "1" && -z "${SM_VLLM_LOAD_FORMAT:-}" ]]; then
  case "${SM_VLLM_MODEL:-${HF_MODEL_ID:-}}" in
  s3://* | gs://* | azure://*) export SM_VLLM_LOAD_FORMAT="runai_streamer" ;;
  esac
fi

PREFIX="SM_VLLM_"
ARG_PREFIX="--"
ARGS=(--port 8080)

# Model auto-detection (when SM_VLLM_MODEL is not provided).
if [[ -z "${SM_VLLM_MODEL:-}" ]]; then
  if [[ -d /opt/ml/model && -n "$(ls -A /opt/ml/model 2>/dev/null)" ]]; then
    echo "INFO: SM_VLLM_MODEL not set, auto-detected model at /opt/ml/model"
    ARGS+=(--model /opt/ml/model)
  elif [[ -n "${HF_MODEL_ID:-}" ]]; then
    echo "INFO: SM_VLLM_MODEL not set, using HF_MODEL_ID=${HF_MODEL_ID}"
    ARGS+=(--model "${HF_MODEL_ID}")
  else
    echo "WARNING: No model specified. Set SM_VLLM_MODEL, HF_MODEL_ID, or mount a model to /opt/ml/model."
  fi
fi

# Map SM_VLLM_* -> --flag [value]; booleans: true=flag only, false=skip.
while IFS='=' read -r key value; do
  arg_name=$(echo "${key#"${PREFIX}"}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
  lower_value=$(echo "${value}" | tr '[:upper:]' '[:lower:]')
  if [[ "${lower_value}" == "true" ]]; then
    ARGS+=("${ARG_PREFIX}${arg_name}")
  elif [[ "${lower_value}" == "false" ]]; then
    continue
  else
    ARGS+=("${ARG_PREFIX}${arg_name}")
    [[ -n "${value}" ]] && ARGS+=("${value}")
  fi
done < <(env | grep "^${PREFIX}")

# SageMaker routing middleware when the base provides it.
if [[ -f /usr/local/bin/sagemaker_serve.py ]]; then
  ARGS+=(--middleware sagemaker_serve.SageMakerRouteMiddleware)
fi

if command -v standard-supervisor >/dev/null 2>&1; then
  exec standard-supervisor python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
fi
exec python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
