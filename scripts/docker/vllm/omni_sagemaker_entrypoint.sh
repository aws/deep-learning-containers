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

# Add SageMaker routing middleware to dispatch /invocations to the correct
# vllm-omni endpoint (e.g. /v1/audio/speech for TTS)
ARGS+=(--middleware omni_sagemaker_serve.SageMakerRouteMiddleware)

exec vllm serve --omni "${ARGS[@]}"
