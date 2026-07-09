#!/usr/bin/env bash
set -e

# Emit telemetry (best-effort, never blocks startup).
bash /usr/local/bin/bash_telemetry.sh >/dev/null 2>&1 || true

# Passive entrypoint for the EC2 (manual) RayTrain image. Unlike the Ray Serve
# image, this does NOT start a Ray cluster — distributed training on EC2 is driven
# by the user, who starts the head and workers themselves, e.g.:
#
#   # on the head node
#   ray start --head --port=6379 --dashboard-host=0.0.0.0
#   # on each worker node
#   ray start --address=<head-ip>:6379
#   # then submit a job
#   ray job submit --address http://<head-ip>:8265 --working-dir . -- python train.py
#
# If a command is passed to `docker run`, exec it; otherwise drop into a shell.
if [ $# -gt 0 ]; then
    exec "$@"
else
    exec /bin/bash
fi
