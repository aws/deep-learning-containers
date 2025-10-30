#!/bin/bash
set -e

curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="/usr/local/bin" sh
uv self update
# uv venv build --python 3.11
# source build/bin/activate
# uv pip install --python 3.11 -r .github/requirements/build.txt
# deactivate
# uv venv test --python 3.11
# source test/bin/activate
# uv pip install --python 3.11 -r .github/requirements/test.txt
# deactivate
# uv venv release --python 3.11
# source release/bin/activate
# uv pip install --python 3.11 -r .github/requirements/release.txt
# deactivate
docker --version