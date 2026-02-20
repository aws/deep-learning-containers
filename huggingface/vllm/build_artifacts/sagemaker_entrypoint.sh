#!/bin/bash
# Check if telemetry file exists before executing
# Execute telemetry script if it exists, suppress errors
bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

# Source CUDA compat for older drivers (e.g., g5 instances)
if command -v nvidia-smi >/dev/null 2>&1 && command -v nvcc >/dev/null 2>&1; then
    source /usr/local/bin/start_cuda_compat.sh
fi

PREFIX="SM_VLLM_"
ARG_PREFIX="--"

ARGS=(--port 8080)

while IFS='=' read -r key value; do
    arg_name=$(echo "${key#"${PREFIX}"}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')

    ARGS+=("${ARG_PREFIX}${arg_name}")
    if [ -n "$value" ]; then
        ARGS+=("$value")
    fi
done < <(env | grep "^${PREFIX}")

exec python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
