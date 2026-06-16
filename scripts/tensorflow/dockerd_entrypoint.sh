#!/bin/bash
# TF training entrypoint (CPU + GPU). Fires telemetry, runs CUDA
# forward-compat shim (no-op on CPU because /usr/local/cuda/compat/libcuda.so.1
# does not exist), then execs the user command.
set -e

# Telemetry — sourced (not exec'd) so it runs in this shell. The telemetry
# script self-checks OPT_OUT_TRACKING and silently skips when set.
if [ -f /usr/local/bin/bash_telemetry.sh ]; then
    source /usr/local/bin/bash_telemetry.sh
fi

# CUDA forward-compatibility (safe no-op on CPU)
bash /usr/local/bin/start_cuda_compat.sh || true

eval "$@"
