#!/bin/bash

# Execute telemetry script if it exists, suppress errors
bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

/usr/local/bin/tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} "$@"
