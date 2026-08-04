#!/usr/bin/env bash
#
# Entrypoint for the HuggingFace vLLM-Omni SageMaker AI target.
#
# Mirrors the AWS omni contract — maps SM_VLLM_* env vars to vLLM CLI flags,
# auto-detects the model from /opt/ml/model or HF_MODEL_ID, and installs the
# /invocations route middleware — and layers HF's run-time defaults on top.
# Anything the user sets via SM_VLLM_* always wins.

# 1. Telemetry (best-effort).
[[ -f /usr/local/bin/bash_telemetry.sh ]] && bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

# 2. HF auto-optimization layer, shared with the HF vLLM DLC.
#
# LMCache defaults OFF here, unlike the vLLM image. LMCache attaches a KV
# connector to an autoregressive KV cache; most omni workloads are diffusion or
# multi-stage pipelines (image, video, audio generation) that have no such cache,
# and AWS's omni entrypoint wires no kv-transfer-config at all. Enabling it by
# default would risk a startup failure for the majority of omni models. Opt in
# with HF_ENABLE_LMCACHE=1 for AR models (e.g. omni chat) that benefit.
: "${HF_ENABLE_LMCACHE:=0}"
export HF_ENABLE_LMCACHE
source /usr/local/bin/hf_optimizations.sh

# LMCache (opt-in): expose the kv-transfer-config through the SM_VLLM_ contract
# so the mapping below turns it into --kv-transfer-config, unless the user set
# one. Passed verbatim — vLLM json.loads() it, so it must not be re-quoted.
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

# Dispatch /invocations to omni routes (/v1/audio/speech, /v1/images/generations,
# /v1/videos, ...) via the X-Amzn-SageMaker-Custom-Attributes route= header.
# Must be the omni middleware, not the vLLM one the base ships at
# /usr/local/bin/sagemaker_serve.py — that router knows nothing about omni routes.
ARGS+=(--middleware omni_sagemaker_serve.SageMakerRouteMiddleware)

# Plain exec, no standard-supervisor wrapper (which the HF vLLM image does use):
# omni serves multi-stage and diffusion models across extra worker processes, and
# supervisord respawn semantics against that process tree are unvalidated. AWS's
# omni entrypoint execs directly for the same reason.
exec vllm serve --omni "${ARGS[@]}"
