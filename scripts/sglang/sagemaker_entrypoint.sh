#!/bin/bash
# Check if telemetry file exists before executing
# Execute telemetry script if it exists, suppress errors
bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

if command -v nvidia-smi >/dev/null 2>&1 && command -v nvcc >/dev/null 2>&1; then
    source /usr/local/bin/start_cuda_compat.sh
fi

# Add nvidia pip package lib dirs to LD_LIBRARY_PATH so that libraries like
# libcusparseLt.so, libcudnn.so, libnccl.so are found at runtime even if
# the ldconfig cache is not available (e.g., SageMaker read-only rootfs).
NVIDIA_PIP_LIBS=$(python3 -c 'import glob; print(":".join(glob.glob("/usr/local/lib*/python3.12/site-packages/nvidia/*/lib")))' 2>/dev/null)
if [ -n "${NVIDIA_PIP_LIBS}" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_PIP_LIBS}:${LD_LIBRARY_PATH}"
fi

echo "Starting server"

PREFIX="SM_SGLANG_"
ARG_PREFIX="--"

ARGS=()

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

echo "Running command: exec python3 -m sglang.launch_server ${ARGS[@]}"
exec python3 -m sglang.launch_server "${ARGS[@]}"
