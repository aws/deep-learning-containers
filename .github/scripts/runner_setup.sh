#!/bin/bash
set -e

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="/usr/local/bin" sh
    uv self update
fi
docker --version
