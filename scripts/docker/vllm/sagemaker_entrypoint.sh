#!/bin/bash
# Check if telemetry file exists before executing
# Execute telemetry script if it exists, suppress errors
bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

# Translate SM_VLLM_* env vars (and the model-source ladder) into CLI arguments.
# The helper emits NUL-delimited tokens, so values containing spaces or newlines stay
# intact and multi-value flags such as --lora-modules get one token per value.
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
