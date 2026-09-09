#!/usr/bin/env bash
set -eo pipefail

if [[ -f /usr/local/bin/start_cuda_compat.sh ]] && command -v nvidia-smi >/dev/null 2>&1; then
  source /usr/local/bin/start_cuda_compat.sh || true
fi

[[ -f /usr/local/bin/bash_telemetry.sh ]] && bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

# Backwards compatibility with the former Hugging Face inference toolkit envs.
if [[ -d "${HF_MODEL_ID:-}" ]]; then
  echo "WARNING: HF_MODEL_ID is a path, please use HF_MODEL_DIR for paths instead."
  export HF_MODEL_DIR="${HF_MODEL_ID}"
  unset HF_MODEL_ID
fi

if [[ -n "${HF_MODEL_DIR:-}" ]]; then
  if [[ -z "${MODEL_DIR:-}" ]]; then
    export MODEL_DIR="${HF_MODEL_DIR}"
  else
    echo "WARNING: MODEL_DIR is already set to '${MODEL_DIR}', keeping its value."
  fi
  unset HF_MODEL_DIR
fi

if [[ -n "${HF_MODEL_ID:-}" && -z "${MODEL_ID:-}" ]]; then
  export MODEL_ID="${HF_MODEL_ID}"
fi

# SageMaker extracts model.tar.gz into /opt/ml/model. Use it only when no model
# was selected through an environment variable or an explicit CLI argument.
model_source_arg=""
for arg in "$@"; do
  case "${arg}" in
    --model-id | --model-id=* | --model-dir | --model-dir=*) model_source_arg="${arg}" ;;
  esac
done

if [[ "${CLOUD:-}" == "sagemaker" \
  && -z "${MODEL_ID:-}" \
  && -z "${MODEL_DIR:-}" \
  && -z "${model_source_arg}" \
  && -d /opt/ml/model \
  && -n "$(find /opt/ml/model -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  export MODEL_DIR=/opt/ml/model
fi

if [[ -n "${HF_TASK:-}" && -z "${TASK:-}" ]]; then
  export TASK="${HF_TASK}"
fi

if [[ -n "${HF_REVISION:-}" && -z "${REVISION:-}" ]]; then
  export REVISION="${HF_REVISION}"
fi

if [[ -n "${HF_TRUST_REMOTE_CODE:-}" && -z "${TRUST_REMOTE_CODE:-}" ]]; then
  export TRUST_REMOTE_CODE="${HF_TRUST_REMOTE_CODE}"
fi

if [[ -n "${MODEL_DIR:-}" ]]; then
  if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "ERROR: Provided MODEL_DIR is not a valid directory" >&2
    exit 1
  fi

  if [[ -f "${MODEL_DIR}/requirements.txt" ]]; then
    echo "INFO: Installing custom dependencies from ${MODEL_DIR}/requirements.txt"
    uv pip install --python "${VIRTUAL_ENV}/bin/python" -r "${MODEL_DIR}/requirements.txt" --no-cache-dir
  fi
fi

if [[ "${1:-}" == "serve" ]]; then
  shift
fi

exec hf-serve "$@"
