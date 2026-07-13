#!/usr/bin/env bash
#
# Entrypoint for the SageMaker training (sagemaker) target.
#
# Layers HF's training performance defaults on top of the AWS base contract,
# then hands off to the stock SageMaker training launcher
# (start_with_right_hostname.sh -> sagemaker-training toolkit), preserving the
# base's multi-node hostname workaround. Anything the user sets in their
# estimator / environment always wins.

# 1. CUDA forward-compat (present on the AWS base; sourced — may export LD_LIBRARY_PATH).
if [[ -f /usr/local/bin/start_cuda_compat.sh ]]; then
  source /usr/local/bin/start_cuda_compat.sh || true
fi

# 2. Telemetry (best-effort; present on the AWS base).
[[ -f /usr/local/bin/bash_telemetry.sh ]] && bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

# 3. HF auto-optimization layer (sets training env defaults).
source /usr/local/bin/hf_optimizations.sh

# 4. Hand off to the stock SageMaker training launcher. Fall back to exec'ing
#    the args directly (e.g. plain `docker run ... python train.py`) when the
#    launcher isn't present.
if command -v start_with_right_hostname.sh >/dev/null 2>&1; then
  exec bash -m start_with_right_hostname.sh "$@"
elif [[ -f /usr/local/bin/start_with_right_hostname.sh ]]; then
  exec bash -m /usr/local/bin/start_with_right_hostname.sh "$@"
fi
exec "$@"
