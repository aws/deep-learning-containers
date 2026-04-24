#!/bin/bash
set -euo pipefail

# Check if telemetry file exists before executing
# Execute telemetry script if it exists, suppress errors
bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

# Source CUDA compat for older drivers (e.g., g5 instances)
if command -v nvidia-smi >/dev/null 2>&1 && command -v nvcc >/dev/null 2>&1; then
    source /usr/local/bin/start_cuda_compat.sh
fi

# SageMaker sends traffic to port 8080 on /ping and /invocations. llama-server
# listens on a loopback-only port; a small Python proxy (llamacpp_sagemaker_serve)
# binds 8080 and forwards to llama-server, similar to vLLM-Omni middleware.
INTERNAL_HOST="${LLAMACPP_SAGEMAKER_INTERNAL_HOST:-127.0.0.1}"
INTERNAL_PORT="${LLAMACPP_SAGEMAKER_INTERNAL_PORT:-8081}"
PROXY_PORT="${LLAMACPP_SAGEMAKER_PROXY_PORT:-8080}"
export LLAMACPP_SAGEMAKER_BACKEND_URL="${LLAMACPP_SAGEMAKER_BACKEND_URL:-http://${INTERNAL_HOST}:${INTERNAL_PORT}}"

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

# Drop any user-supplied --host / --port so inference stays on the internal bind.
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
ARGS+=(--host "$INTERNAL_HOST" --port "$INTERNAL_PORT")

echo "[sagemaker] llama-server args: ${ARGS[*]}" >&2

/app/llama-server "${ARGS[@]}" &
LLAMA_PID=$!

wait_for_llama() {
    local i
    for i in $(seq 1 120); do
        if curl -sf "http://${INTERNAL_HOST}:${INTERNAL_PORT}/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

if ! wait_for_llama; then
    echo "[sagemaker] llama-server did not become healthy on ${INTERNAL_HOST}:${INTERNAL_PORT}" >&2
    kill -TERM "$LLAMA_PID" 2>/dev/null || true
    wait "$LLAMA_PID" 2>/dev/null || true
    exit 1
fi

shutdown() {
    kill -TERM "$UVICORN_PID" 2>/dev/null || true
    kill -TERM "$LLAMA_PID" 2>/dev/null || true
    wait "$UVICORN_PID" 2>/dev/null || true
    wait "$LLAMA_PID" 2>/dev/null || true
}

trap shutdown SIGTERM SIGINT

if [ -n "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="${PYTHONPATH}:/usr/local/lib/llamacpp_sagemaker"
else
    export PYTHONPATH="/usr/local/lib/llamacpp_sagemaker"
fi
python3 -m uvicorn llamacpp_sagemaker_serve:app --host 0.0.0.0 --port "$PROXY_PORT" --log-level info &
UVICORN_PID=$!

wait "$UVICORN_PID"
exit_code=$?
shutdown
exit "$exit_code"
