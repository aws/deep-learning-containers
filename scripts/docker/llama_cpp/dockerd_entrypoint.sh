#!/usr/bin/env bash
# EC2 entrypoint for the llama.cpp DLC.
#
# SECURITY: llama-server exposes an OpenAI-compatible HTTP API on 0.0.0.0:8080.
# The endpoint is UNAUTHENTICATED unless an API key is supplied. Set LLAMA_API_KEY
# (or pass --api-key via container args) to require a bearer token. Regardless of
# auth, the endpoint MUST be network-isolated on any deployment target: use
# restrictive security groups and/or an auth+TLS reverse proxy. Do not expose it
# directly to untrusted networks.
#
# Emits telemetry (best-effort) then launches llama-server with the passed args.
bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

API_KEY_ARG=()
if [ -n "${LLAMA_API_KEY:-}" ]; then
  API_KEY_ARG=(--api-key "${LLAMA_API_KEY}")
fi

exec llama-server --host 0.0.0.0 --port 8080 "${API_KEY_ARG[@]}" "$@"
