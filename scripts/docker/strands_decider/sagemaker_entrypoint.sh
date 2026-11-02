#!/bin/bash
# SageMaker runs `docker run IMAGE serve`; the `serve` argument is ignored.
# strands-decider serves GET /ping and POST /invocations (and /health, /v1/systemone) on port 8080, after the
# warm-up and with the GPU options of sagemaker_serve.py (DECIDER_WARMUP, DECIDER_DEVICE_MAP).
set -e

# shellcheck disable=SC2086 # DECIDER_ARGS holds several options
exec python3 /usr/local/bin/sagemaker_serve.py serve "${DECIDER_CHECKPOINT}" --host 0.0.0.0 --port 8080 --device "${DECIDER_DEVICE}" ${DECIDER_ARGS}
