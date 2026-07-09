#!/bin/bash
# Check if telemetry file exists before executing
# Execute telemetry script if it exists, suppress errors
bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

if command -v nvidia-smi >/dev/null 2>&1 && command -v nvcc >/dev/null 2>&1; then
    bash /usr/local/bin/start_cuda_compat.sh
fi

echo "Starting server"

PREFIX="SM_SGLANG_"
ARG_PREFIX="--"

# Engine selector (default: llm). Set SM_SGLANG_ENGINE=diffusion to serve a
# FLUX.2 / diffusion pipeline via sglang.multimodal_gen instead of the LLM
# engine. This var controls the launch module and is NOT forwarded as a flag.
ENGINE=$(echo "${SM_SGLANG_ENGINE:-llm}" | tr '[:upper:]' '[:lower:]')

ARGS=()

while IFS='=' read -r key value; do
    # SM_SGLANG_ENGINE selects the launch module; it is not a server flag.
    if [ "$key" = "${PREFIX}ENGINE" ]; then
        continue
    fi

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

# Add default port only if not already set
if ! [[ " ${ARGS[@]} " =~ " --port " ]]; then
    ARGS+=(--port "${SM_SGLANG_PORT:-8080}")
fi

# Add default host only if not already set
if ! [[ " ${ARGS[@]} " =~ " --host " ]]; then
    ARGS+=(--host "${SM_SGLANG_HOST:-0.0.0.0}")
fi

# Add default model-path only if not already set
if ! [[ " ${ARGS[@]} " =~ " --model-path " ]]; then
    ARGS+=(--model-path "${SM_SGLANG_MODEL_PATH:-/opt/ml/model}")
fi

if [ "$ENGINE" = "diffusion" ]; then
    LAUNCH_MODULE="sglang.multimodal_gen.runtime.launch_server"
else
    LAUNCH_MODULE="sglang.launch_server"
fi

echo "Running command: exec python3 -m ${LAUNCH_MODULE} ${ARGS[@]}"
exec python3 -m "${LAUNCH_MODULE}" "${ARGS[@]}"
