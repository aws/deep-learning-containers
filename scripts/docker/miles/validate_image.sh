#!/usr/bin/env bash
# Build-time gate shared by every Miles image target. Run it as the last step
# of a target, after every package that target ships is installed.
set -euo pipefail

DLC_DIR=/opt/aws/dlc

ffmpeg -version
python "${DLC_DIR}/validate_cuda_jit_toolchain.py"
python "${DLC_DIR}/validate_wheel_artifacts.py" --source-manifest "${DLC_DIR}/SOURCE_MANIFEST"
python "${DLC_DIR}/validate_uv_lock.py"
python "${DLC_DIR}/validate_pip_check.py" "${DLC_DIR}/PIP_CHECK_ALLOWLIST"
python "${DLC_DIR}/smoke_test.py"
python -m pip freeze >"${DLC_DIR}/PYTHON_PACKAGES_LOCK"
