#!/bin/bash
set -e

curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="/usr/local/bin" sh
uv self update
docker --version
