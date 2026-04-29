#!/bin/bash
set -euo pipefail

# Check if telemetry file exists before executing
# Execute telemetry script if it exists, suppress errors
bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

# Source CUDA compat for older drivers (e.g., g5 instances)
if [ -f /usr/local/bin/start_cuda_compat.sh ] \
    && command -v nvidia-smi >/dev/null 2>&1 \
    && command -v nvcc >/dev/null 2>&1; then
    source /usr/local/bin/start_cuda_compat.sh
fi

# SageMaker sends traffic to port 8080 on /ping and /invocations. The custom
# llama-server build handles those routes directly.
HOST="${LLAMACPP_SAGEMAKER_HOST:-0.0.0.0}"
PORT="${SAGEMAKER_BIND_TO_PORT:-${LLAMACPP_SAGEMAKER_PORT:-8080}}"

PREFIX="SM_LLAMACPP_"
ARG_PREFIX="--"

ARGS=()

while IFS='=' read -r key value; do
    arg_name=$(echo "${key#"${PREFIX}"}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')

    ARGS+=("${ARG_PREFIX}${arg_name}")
    if [ -n "$value" ]; then
        ARGS+=("$value")
    fi
done < <(env | grep "^${PREFIX}" || true)

# Drop any user-supplied --host / --port so SageMaker can always reach the server.
normalized=()
skip_next=0
for a in "${ARGS[@]}"; do
    if [ "$skip_next" -eq 1 ]; then
        skip_next=0
        continue
    fi
    if [ "$a" = "--host" ] || [ "$a" = "--port" ]; then
        skip_next=1
        continue
    fi
    normalized+=("$a")
done
ARGS=("${normalized[@]}")
ARGS+=(--host "$HOST" --port "$PORT")

echo "[sagemaker] llama-server args: ${ARGS[*]}" >&2

exec /app/llama-server "${ARGS[@]}"
