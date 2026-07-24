#!/usr/bin/env bash
#
# Entrypoint for the EC2 / EKS (ec2) target.
#
# Applies HF's training performance defaults, then hands off to the AWS base
# entrypoint (CUDA forward-compat, telemetry, multi-node SSH) so plain
# `docker run`/`torchrun` and multi-node EC2 training behave exactly as on the
# stock DLC. Anything the user sets always wins; defaults to an interactive shell.
#
#   docker run --gpus all hf-pytorch-training:ec2 \
#     torchrun --nproc-per-node 8 train.py --model <hf-id>

# HF auto-optimization layer (sets training env defaults).
source /usr/local/bin/hf_optimizations.sh

# Default command when none is given.
if [[ "$#" -eq 0 ]]; then
  set -- /bin/bash
fi

# Delegate to the stock AWS base entrypoint when present (it owns CUDA
# forward-compat + telemetry). Our script is ec2_entrypoint.sh, so the base's
# /usr/local/bin/entrypoint.sh is untouched and safe to chain to.
if [[ -x /usr/local/bin/entrypoint.sh ]]; then
  exec /usr/local/bin/entrypoint.sh "$@"
fi

# Fallback: replicate the essentials ourselves.
if [[ -f /usr/local/bin/start_cuda_compat.sh ]]; then
  source /usr/local/bin/start_cuda_compat.sh || true
fi
[[ -f /usr/local/bin/bash_telemetry.sh ]] && bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true
exec "$@"
