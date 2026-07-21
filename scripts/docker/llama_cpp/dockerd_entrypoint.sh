#!/usr/bin/env bash
# EC2 entrypoint for the llama.cpp ARM64 DLC.
# Emits telemetry (best-effort) then launches llama-server with the passed args.
bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

exec llama-server --host 0.0.0.0 --port 8080 "$@"
