#!/usr/bin/env bash
# Check if telemetry file exists before executing
# Execute telemetry script if it exists, suppress errors
bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

# Add nvidia pip package lib paths to LD_LIBRARY_PATH (fallback for --gpus=all
# environments where nvidia-container-toolkit may overwrite the ldconfig cache)
if [ -f /etc/nvidia-pip-lib-paths ]; then
    export LD_LIBRARY_PATH="$(cat /etc/nvidia-pip-lib-paths):${LD_LIBRARY_PATH}"
fi

python3 -m sglang.launch_server "$@"
