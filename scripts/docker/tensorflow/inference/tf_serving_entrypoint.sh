#!/bin/bash

# Execute telemetry script if it exists, suppress errors
bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

# exec so tensorflow_model_server replaces this shell as the container's PID —
# lets docker's SIGTERM propagate directly to TFS for graceful shutdown
# (SageMaker endpoint teardown / customer autoscaling). Without exec, the wrapping
# bash owns PID 1, may not forward SIGTERM, and docker SIGKILLs after 10s.
exec /usr/local/bin/tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} "$@"
