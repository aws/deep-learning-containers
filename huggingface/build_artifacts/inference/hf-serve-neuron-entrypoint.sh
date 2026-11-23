#!/usr/bin/env bash
# Entrypoint for the Hugging Face PyTorch NeuronX inference DLC (Torch NeuronX 2.13+).
# Maps the former Hugging Face inference toolkit / SageMaker environment variables
# to hf-serve, which serves on the `neuron` device by default.
set -eo pipefail

# Optional CloudWatch Neuron monitor (enabled through NEURON_MONITOR_CW_REGION).
[[ -x /usr/local/bin/neuron-monitor.sh ]] && /usr/local/bin/neuron-monitor.sh

if [[ "${1:-}" == "serve" ]]; then
  shift
elif [[ $# -gt 0 && "${1:-}" != -* ]]; then
  # Any other command runs as is, e.g. `docker run <image> python3 script.py`.
  exec "$@"
fi

ARGS=(
  --host "${SM_HF_SERVE_HOST:-${HOST:-0.0.0.0}}"
  --port "${SAGEMAKER_BIND_TO_PORT:-${SM_HF_SERVE_PORT:-${PORT:-8080}}}"
)

# SM_HF_SERVE_<NAME>=<value> becomes `--<name> <value>`; `true` becomes a flag.
PREFIX="SM_HF_SERVE_"
while IFS='=' read -r key value; do
  case "${key}" in
    SM_HF_SERVE_HOST | SM_HF_SERVE_PORT) continue ;;
  esac
  arg_name=$(echo "${key#"${PREFIX}"}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
  lower_value=$(echo "${value}" | tr '[:upper:]' '[:lower:]')
  if [[ "${lower_value}" == "true" ]]; then
    ARGS+=("--${arg_name}")
  elif [[ "${lower_value}" != "false" ]]; then
    ARGS+=("--${arg_name}")
    [[ -n "${value}" ]] && ARGS+=("${value}")
  fi
done < <(env | grep "^${PREFIX}" || true)

# Backwards compatibility with the former Hugging Face inference toolkit envs.
if [[ -d "${HF_MODEL_ID:-}" ]]; then
  echo "WARNING: HF_MODEL_ID is a path, please use HF_MODEL_DIR for paths instead."
  export HF_MODEL_DIR="${HF_MODEL_ID}"
  unset HF_MODEL_ID
fi
if [[ -n "${HF_MODEL_DIR:-}" && -z "${MODEL_DIR:-}" ]]; then
  export MODEL_DIR="${HF_MODEL_DIR}"
fi
unset HF_MODEL_DIR
if [[ -n "${HF_MODEL_ID:-}" && -z "${MODEL_ID:-}" ]]; then
  export MODEL_ID="${HF_MODEL_ID}"
fi
[[ -n "${HF_TASK:-}" && -z "${TASK:-}" ]] && export TASK="${HF_TASK}"
[[ -n "${HF_REVISION:-}" && -z "${REVISION:-}" ]] && export REVISION="${HF_REVISION}"
[[ -n "${HF_TRUST_REMOTE_CODE:-}" && -z "${TRUST_REMOTE_CODE:-}" ]] && export TRUST_REMOTE_CODE="${HF_TRUST_REMOTE_CODE}"

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

if [[ -n "${MODEL_DIR:-}" ]]; then
  if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "ERROR: Provided MODEL_DIR is not a valid directory" >&2
    exit 1
  fi
  # A model archive with a custom handler (at its root, or under code/ as packed
  # by the SageMaker SDK `entry_point`) serves through it unless a task is set.
  # hf-serve still requires TRUST_REMOTE_CODE=true to run the handler.
  handler_file="${CUSTOM_HANDLER_FILE:-handler.py}"
  if [[ -z "${TASK:-}" ]]; then
    if [[ -f "${MODEL_DIR}/${handler_file}" ]]; then
      export TASK=custom
    elif [[ -f "${MODEL_DIR}/code/${handler_file}" ]]; then
      export TASK=custom CUSTOM_HANDLER_FILE="code/${handler_file}"
    fi
  fi
  for requirements in "${MODEL_DIR}/requirements.txt" "${MODEL_DIR}/code/requirements.txt"; do
    if [[ -f "${requirements}" ]]; then
      echo "INFO: Installing custom dependencies from ${requirements}"
      python3 -m pip install --no-cache-dir -r "${requirements}"
    fi
  done
fi

# hf-serve selects the `neuron` device with the default `--device auto`.
exec hf-serve "${ARGS[@]}" "$@"
