#!/usr/bin/env bash
set -e

# CUDA forward compatibility (sourced so its LD_LIBRARY_PATH export persists).
if [ -f /usr/local/bin/start_cuda_compat.sh ]; then
    source /usr/local/bin/start_cuda_compat.sh
fi

# Emit telemetry (best-effort, never blocks startup).
bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

# Passive entrypoint: user drives ray start; exec any passed command, else a shell.
if [ $# -gt 0 ]; then
    exec "$@"
else
    exec /bin/bash
fi
