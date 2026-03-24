#!/bin/bash
set -euo pipefail

# llama.cpp Health Check Test
# Starts llama-server without a model and verifies the /health endpoint responds.
# Usage: llamacpp_health_check_test.sh

LLAMACPP_PORT=8080
HEALTH_TIMEOUT=30
HEALTH_INTERVAL=2

echo "=== Starting llama-server (no model, health check only) ==="
llama-server \
  --port "${LLAMACPP_PORT}" \
  --host 0.0.0.0 &
SERVER_PID=$!

cleanup() {
  echo "=== Stopping llama-server ==="
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Waiting for /health endpoint ==="
elapsed=0
while [ "${elapsed}" -lt "${HEALTH_TIMEOUT}" ]; do
  if curl -sf http://localhost:${LLAMACPP_PORT}/health >/dev/null 2>&1; then
    echo "Health endpoint responded after ${elapsed}s"
    echo "=== PASSED: llama-server health check ==="
    exit 0
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "ERROR: llama-server process died"
    exit 1
  fi
  sleep "${HEALTH_INTERVAL}"
  elapsed=$((elapsed + HEALTH_INTERVAL))
done

echo "ERROR: Health check timed out after ${HEALTH_TIMEOUT}s"
exit 1
