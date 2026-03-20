#!/usr/bin/env bash
# Check if telemetry file exists before executing
# Execute telemetry script if it exists, suppress errors
bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

# Add nvidia pip package lib dirs to LD_LIBRARY_PATH so that libraries like
# libcusparseLt.so, libcudnn.so, libnccl.so are found at runtime.
NVIDIA_PIP_LIBS=$(python3 -c 'import glob; print(":".join(glob.glob("/usr/local/lib*/python3.12/site-packages/nvidia/*/lib")))' 2>/dev/null)
if [ -n "${NVIDIA_PIP_LIBS}" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_PIP_LIBS}:${LD_LIBRARY_PATH}"
fi

python3 -m sglang.launch_server "$@"
