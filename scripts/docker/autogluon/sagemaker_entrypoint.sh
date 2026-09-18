#!/usr/bin/env bash
# Unified AutoGluon SageMaker entrypoint. SageMaker invokes training containers
# with `train` and inference containers with `serve`.
set -euo pipefail

case "${1:-}" in
  train)
    # Preserve the PyTorch training DLC's hostname workaround, CUDA setup, and
    # SageMaker training-toolkit launcher.
    cd /
    exec bash -m start_with_right_hostname.sh "$@"
    ;;
  serve)
    shift
    # Serving does not start a login shell, so fire telemetry explicitly.
    bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true
    # Install model-artifact dependencies once, before Gunicorn forks workers.
    /opt/autogluon-server/install_requirements.py
    # Delegate CUDA forward compatibility to the inherited generic entrypoint.
    exec /usr/local/bin/entrypoint.sh gunicorn \
      --config /opt/autogluon-server/gunicorn.conf.py \
      --chdir /opt/autogluon-server \
      "$@" \
      server:app
    ;;
  *)
    exec /usr/local/bin/entrypoint.sh "$@"
    ;;
esac
