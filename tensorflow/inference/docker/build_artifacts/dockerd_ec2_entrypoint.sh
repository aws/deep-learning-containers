#!/usr/bin/env bash

# Execute telemetry script if it exists, suppress errors
bash /usr/local/bin/bash_telemetry.sh 2>/dev/null || true

eval '"$@"'