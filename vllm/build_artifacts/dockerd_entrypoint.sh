#!/usr/bin/env bash
# Check if telemetry file exists before executing
# Execute telemetry script if it exists, suppress errors
bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

python3 -m vllm.entrypoints.openai.api_server "$@"