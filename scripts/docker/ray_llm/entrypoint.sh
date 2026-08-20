#!/bin/bash
set -e

source /usr/local/bin/start_cuda_compat.sh

exec "$@"
