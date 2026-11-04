#!/bin/bash
# SageMaker runs `docker run IMAGE serve`; any other command runs as given (e.g. `bash`).
# strands-decider serves GET /ping and POST /invocations (and /health, /v1/systemone) on port 8080; /ping fails
# until the model is loaded and warmed up. The environment maps to `strands-decider serve` options:
#   DECIDER_CHECKPOINT  checkpoint folder. Default: SM_MODEL_DIR (/opt/ml/model, where SageMaker puts the model)
#                       if it holds the checkpoint, else its one subfolder that does (e.g. <model>/ next to base/)
#   DECIDER_DEVICE      cuda (default) or cpu
#   DECIDER_WARMUP      0 skips the warm-up (--no-warmup)
#   DECIDER_DEVICE_MAP  e.g. auto: split a model larger than one GPU across the visible GPUs (--device-map)
#   DECIDER_ARGS        any other options, e.g. "--max-batch 8"
set -e

if [ $# -gt 0 ] && [ "$1" != "serve" ]; then
  exec "$@"
fi

# CUDA forward compatibility, for hosts whose driver is older than the image's CUDA.
if [ -f /usr/local/bin/start_cuda_compat.sh ]; then
  source /usr/local/bin/start_cuda_compat.sh >&2 || true
fi
# Serving does not start a login shell, so send telemetry explicitly.
bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

is_checkpoint() {
  [ -f "$1/strands_decider_config.json" ] || [ -f "$1/hobson_config.json" ]
}
if [ -z "${DECIDER_CHECKPOINT:-}" ]; then
  model_dir=${SM_MODEL_DIR:-/opt/ml/model}
  if is_checkpoint "${model_dir}"; then
    DECIDER_CHECKPOINT=${model_dir}
  else
    found=()
    for dir in "${model_dir}"/*/; do
      if is_checkpoint "${dir%/}"; then found+=("${dir%/}"); fi
    done
    if [ ${#found[@]} -ne 1 ]; then
      echo "No single strands-decider checkpoint in ${model_dir} (found ${#found[@]}); set DECIDER_CHECKPOINT" >&2
      exit 1
    fi
    DECIDER_CHECKPOINT=${found[0]}
  fi
fi

args=()
if [ "${DECIDER_WARMUP:-1}" = "0" ]; then
  args+=(--no-warmup)
fi
if [ -n "${DECIDER_DEVICE_MAP:-}" ]; then
  args+=(--device-map "${DECIDER_DEVICE_MAP}")
fi

# DECIDER_ARGS holds several options: split it on spaces, without expanding glob characters.
set -f
# shellcheck disable=SC2086
exec strands-decider serve "${DECIDER_CHECKPOINT}" --host 0.0.0.0 --port 8080 --device "${DECIDER_DEVICE:-cuda}" \
  "${args[@]}" ${DECIDER_ARGS}
