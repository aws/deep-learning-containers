#!/bin/bash
# Check if telemetry file exists before executing
# Execute telemetry script if it exists, suppress errors
bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

PREFIX="SM_LLAMACPP_"

ARGS=(--port 8080 --host 0.0.0.0)

# Auto-detect model if SM_LLAMACPP_MODEL is not set
if [ -z "${SM_LLAMACPP_MODEL}" ]; then
    if [ -d "/opt/ml/model" ] && [ "$(ls -A /opt/ml/model 2>/dev/null)" ]; then
        # Find the first GGUF file in the model directory
        GGUF_FILE=$(find /opt/ml/model -name "*.gguf" -type f | head -1)
        if [ -n "${GGUF_FILE}" ]; then
            echo "INFO: SM_LLAMACPP_MODEL not set, auto-detected GGUF model at ${GGUF_FILE}"
            ARGS+=(--model "${GGUF_FILE}")
        else
            echo "WARNING: /opt/ml/model exists but no .gguf file found."
        fi
    elif [ -n "${HF_MODEL_ID}" ]; then
        echo "INFO: SM_LLAMACPP_MODEL not set, using HF_MODEL_ID=${HF_MODEL_ID}"
        ARGS+=(--model "${HF_MODEL_ID}")
    else
        echo "WARNING: No model specified. Set SM_LLAMACPP_MODEL, HF_MODEL_ID, or mount a model to /opt/ml/model."
    fi
fi

# Map of SM_LLAMACPP_ env vars that use short-form llama-server flags
declare -A SHORT_FLAGS=(
    ["CTX_SIZE"]="-c"
    ["N_GPU_LAYERS"]="-ngl"
    ["PARALLEL"]="-np"
)

while IFS='=' read -r key value; do
    suffix="${key#"${PREFIX}"}"

    # Use short flag if defined, otherwise convert to --long-form
    if [ -n "${SHORT_FLAGS[$suffix]+x}" ]; then
        arg_name="${SHORT_FLAGS[$suffix]}"
    else
        arg_name="--$(echo "${suffix}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')"
    fi

    # Handle boolean flags: true -> flag only, false -> skip entirely
    lower_value=$(echo "$value" | tr '[:upper:]' '[:lower:]')
    if [ "$lower_value" = "true" ]; then
        ARGS+=("${arg_name}")
    elif [ "$lower_value" = "false" ]; then
        continue
    else
        ARGS+=("${arg_name}")
        if [ -n "$value" ]; then
            ARGS+=("$value")
        fi
    fi
done < <(env | grep "^${PREFIX}")

echo "Running command: exec llama-server ${ARGS[*]}"
exec llama-server "${ARGS[@]}"
