#!/bin/bash
# SageMaker runs `docker run IMAGE serve`; the `serve` argument is ignored.
# strands-decider serves GET /ping and POST /invocations (and /health, /v1/systemone) on port 8080.
set -e

# shellcheck disable=SC2086 # DECIDER_ARGS holds several options
exec strands-decider serve "${DECIDER_CHECKPOINT}" --host 0.0.0.0 --port 8080 --device "${DECIDER_DEVICE}" ${DECIDER_ARGS}
