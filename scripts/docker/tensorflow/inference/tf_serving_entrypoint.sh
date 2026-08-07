#!/bin/bash

# Best-effort telemetry.
bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

# exec so TFS replaces this shell as PID 1 — SIGTERM reaches TFS directly for
# graceful shutdown; otherwise bash owns PID 1 and docker SIGKILLs after 10s.
exec /usr/local/bin/tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} "$@"
