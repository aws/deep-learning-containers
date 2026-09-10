#!/bin/bash
# SageMaker entrypoint: CPU env defaults + nobind, then the shared vLLM middleware path.
bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

source /usr/local/bin/vllm_cpu_env.sh

export VLLM_CPU_OMP_THREADS_BIND="${VLLM_CPU_OMP_THREADS_BIND:-nobind}"

ARGS_FILE=$(mktemp)
trap 'rm -f "${ARGS_FILE}"' EXIT
if ! python3 /usr/local/bin/sagemaker_args.py >"${ARGS_FILE}"; then
    echo "ERROR: failed to build vLLM arguments from SM_VLLM_* environment variables" >&2
    exit 1
fi
ARGS=()
while IFS= read -r -d '' token; do
    ARGS+=("${token}")
done <"${ARGS_FILE}"
rm -f "${ARGS_FILE}"
trap - EXIT

# Add SageMaker routing middleware when available (amzn2023 image).
if [ -f "/usr/local/bin/sagemaker_serve.py" ]; then
    ARGS+=(--middleware sagemaker_serve.SageMakerRouteMiddleware)
fi

exec standard-supervisor python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
