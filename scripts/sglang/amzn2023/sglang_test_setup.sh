#!/bin/bash
set -eux

# Use --system when not in a virtualenv (Ubuntu image), omit when venv is active (AL2023)
UV_FLAGS=""
if [ -z "${VIRTUAL_ENV:-}" ]; then
  UV_FLAGS="--system"
fi

# Install SGLang test dependencies
uv pip install $UV_FLAGS pytest pytest-asyncio
uv pip install $UV_FLAGS hf_transfer
