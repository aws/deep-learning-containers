#!/bin/bash
# Check if telemetry file exists before executing
# Execute telemetry script if it exists, suppress errors
bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

PREFIX="SM_VLLM_"
ARG_PREFIX="--"

# vllm-omni listens on 8081; the serve proxy on 8080 (SageMaker's port)
ARGS=(--port 8081)

# Auto-detect model if SM_VLLM_MODEL is not set
if [ -z "${SM_VLLM_MODEL}" ]; then
    if [ -d "/opt/ml/model" ] && [ "$(ls -A /opt/ml/model 2>/dev/null)" ]; then
        echo "INFO: SM_VLLM_MODEL not set, auto-detected model at /opt/ml/model"
        ARGS+=(--model /opt/ml/model)
    elif [ -n "${HF_MODEL_ID}" ]; then
        echo "INFO: SM_VLLM_MODEL not set, using HF_MODEL_ID=${HF_MODEL_ID}"
        ARGS+=(--model "${HF_MODEL_ID}")
    else
        echo "WARNING: No model specified. Set SM_VLLM_MODEL, HF_MODEL_ID, or mount a model to /opt/ml/model."
    fi
fi

while IFS='=' read -r key value; do
    arg_name=$(echo "${key#"${PREFIX}"}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')

    # Handle boolean flags: true -> flag only, false -> skip entirely
    lower_value=$(echo "$value" | tr '[:upper:]' '[:lower:]')
    if [ "$lower_value" = "true" ]; then
        ARGS+=("${ARG_PREFIX}${arg_name}")
    elif [ "$lower_value" = "false" ]; then
        continue
    else
        ARGS+=("${ARG_PREFIX}${arg_name}")
        if [ -n "$value" ]; then
            ARGS+=("$value")
        fi
    fi
done < <(env | grep "^${PREFIX}")

# Start vllm-omni on port 8081 in background
vllm serve --omni "${ARGS[@]}" &
VLLM_PID=$!

# Start the SageMaker serve proxy on port 8080 (foreground)
exec python3 /usr/local/bin/omni_sagemaker_serve.py
