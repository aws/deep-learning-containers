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
# standard-supervisor (the default, PROCESS_AUTO_RECOVERY=true) flattens argv
# into a single string that supervisord re-parses with shlex — which strips the
# JSON's double quotes and makes vLLM fail with "key must be a string". We
# single-quote-wrap the value so shlex hands the clean JSON back. In direct mode
# (PROCESS_AUTO_RECOVERY=false) argv is exec'd verbatim, so pass raw JSON.
if [[ -n "${HF_LMCACHE_KV_CONFIG:-}" && -z "${SM_VLLM_KV_TRANSFER_CONFIG:-}" ]]; then
  _auto_recovery="$(echo "${PROCESS_AUTO_RECOVERY:-true}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${_auto_recovery}" == "true" || "${_auto_recovery}" == "1" ]]; then
    export SM_VLLM_KV_TRANSFER_CONFIG="'${HF_LMCACHE_KV_CONFIG}'"
  else
    export SM_VLLM_KV_TRANSFER_CONFIG="${HF_LMCACHE_KV_CONFIG}"
  fi
  unset _auto_recovery
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
