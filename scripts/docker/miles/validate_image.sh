#!/usr/bin/env bash
# Build-time integrity gate shared by every Miles image target: the image must
# come from the official wheels and the uv lock. Run it as the last step of a
# target, after every package that target ships is installed. Behavior checks
# live in test/miles/unit/ and run against the built image.
set -euo pipefail

DLC_DIR=/opt/aws/dlc

python "${DLC_DIR}/validate_wheel_artifacts.py" --source-manifest "${DLC_DIR}/SOURCE_MANIFEST"
python "${DLC_DIR}/validate_uv_lock.py"
python "${DLC_DIR}/validate_pip_check.py" "${DLC_DIR}/PIP_CHECK_ALLOWLIST"
python -m pip freeze >"${DLC_DIR}/PYTHON_PACKAGES_LOCK"
