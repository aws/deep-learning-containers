#!/usr/bin/env bash
# SageMaker entrypoint: llama-server binds 127.0.0.1:8081, nginx serves :8080 and
# maps GET /ping -> /health, POST /invocations -> /v1/chat/completions.
# Server flags come from SM_LLAMA_CPP_* env vars (lowercased, _ -> -), e.g.
# SM_LLAMA_CPP_CTX_SIZE=4096 -> --ctx-size 4096; SM_LLAMA_CPP_VERBOSE=true -> --verbose.
set -euo pipefail

bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

MODEL_DIR="${SM_LLAMA_CPP_MODEL_DIR:-/opt/ml/model}"
UPSTREAM_PORT="${LLAMA_CPP_UPSTREAM_PORT:-8081}"
SERVE_PORT="${SM_LLAMA_CPP_PORT:-8080}"

PREFIX="SM_LLAMA_CPP_"
ARGS=()

# Resolve the model: honor an explicit SM_LLAMA_CPP_MODEL, else pick the first
# .gguf under the SageMaker model dir. Pure-shell glob (2 levels deep) — the
# slim runtime image ships no `find`.
if [ -n "${SM_LLAMA_CPP_MODEL:-}" ]; then
  ARGS+=(--model "${SM_LLAMA_CPP_MODEL}")
else
  gguf_file=""
  shopt -s nullglob 2>/dev/null || true
  for candidate in "${MODEL_DIR}"/*.gguf "${MODEL_DIR}"/*/*.gguf; do
    [ -f "${candidate}" ] && { gguf_file="${candidate}"; break; }
  done
  if [ -n "${gguf_file}" ]; then
    echo "INFO: auto-detected GGUF model ${gguf_file}"
    ARGS+=(--model "${gguf_file}")
  else
    echo "WARNING: no *.gguf found under ${MODEL_DIR}; set SM_LLAMA_CPP_MODEL or mount a GGUF."
  fi
fi

# Translate remaining SM_LLAMA_CPP_* env vars into llama-server flags.
while IFS='=' read -r key value; do
  case "${key}" in
    "${PREFIX}MODEL"|"${PREFIX}MODEL_DIR"|"${PREFIX}PORT") continue ;;
  esac
  arg_name="$(echo "${key#"${PREFIX}"}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')"
  lower_value="$(echo "${value}" | tr '[:upper:]' '[:lower:]')"
  if [ "${lower_value}" = "true" ]; then
    ARGS+=("--${arg_name}")
  elif [ "${lower_value}" = "false" ]; then
    continue
  else
    ARGS+=("--${arg_name}")
    [ -n "${value}" ] && ARGS+=("${value}")
  fi
done < <(env | grep "^${PREFIX}" || true)

echo "Starting llama-server on 127.0.0.1:${UPSTREAM_PORT}"
llama-server --host 127.0.0.1 --port "${UPSTREAM_PORT}" "${ARGS[@]}" &
LLAMA_PID=$!

cleanup() { kill "${LLAMA_PID}" 2>/dev/null || true; }
trap cleanup EXIT

# Render the nginx config with the resolved ports, then run nginx in foreground.
export UPSTREAM_PORT SERVE_PORT
envsubst '${UPSTREAM_PORT} ${SERVE_PORT}' \
  </etc/nginx/llama_cpp_sagemaker.conf.template \
  >/etc/nginx/conf.d/llama_cpp_sagemaker.conf
rm -f /etc/nginx/conf.d/default.conf 2>/dev/null || true

echo "Starting nginx SageMaker proxy on 0.0.0.0:${SERVE_PORT}"
exec nginx -g 'daemon off;'
