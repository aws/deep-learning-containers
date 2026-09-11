#!/usr/bin/env bash
set -eo pipefail

if [[ "${1:-}" == "serve" ]]; then
  shift
fi

ARGS=(
  --host "${SM_HF_SERVE_HOST:-${HOST:-0.0.0.0}}"
  --port "${SM_HF_SERVE_PORT:-${PORT:-8080}}"
)
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

# CLOUD=sagemaker enables hf-serve's native /ping and /invocations middleware.
exec /usr/local/bin/hf_serve_entrypoint.sh "${ARGS[@]}" "$@"
