#!/bin/bash
# SageMaker runs `docker run IMAGE serve`; the `serve` argument is ignored.
# strands-decider serves GET /ping and POST /invocations (and /health, /v1/systemone) on port 8080; /ping fails
# until the model is loaded and warmed up. The environment maps to `strands-decider serve` options:
#   DECIDER_CHECKPOINT  model folder (SageMaker copies the model to /opt/ml/model)
#   DECIDER_DEVICE      cuda (default) or cpu
#   DECIDER_WARMUP      0 skips the warm-up (--no-warmup)
#   DECIDER_DEVICE_MAP  e.g. auto: split a model larger than one GPU across the visible GPUs (--device-map)
#   DECIDER_ARGS        any other options, e.g. "--max-batch 8"
set -e

args=()
if [ "${DECIDER_WARMUP:-1}" = "0" ]; then
  args+=(--no-warmup)
fi
if [ -n "${DECIDER_DEVICE_MAP:-}" ]; then
  args+=(--device-map "${DECIDER_DEVICE_MAP}")
fi

# shellcheck disable=SC2086 # DECIDER_ARGS holds several options
exec strands-decider serve "${DECIDER_CHECKPOINT}" --host 0.0.0.0 --port 8080 --device "${DECIDER_DEVICE}" \
  "${args[@]}" ${DECIDER_ARGS}
