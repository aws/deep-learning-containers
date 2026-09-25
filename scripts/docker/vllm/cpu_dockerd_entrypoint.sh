#!/usr/bin/env bash
# EC2 entrypoint: telemetry + CPU env defaults, then vllm serve.
bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

source /usr/local/bin/vllm_cpu_env.sh

vllm serve "$@"
