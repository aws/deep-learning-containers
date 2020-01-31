#!/bin/bash
#
# Build all the docker images.

set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

${DIR}/build.sh --version 1.14.0 --arch eia
${DIR}/build.sh --version 1.15.0 --arch cpu
${DIR}/build.sh --version 1.15.0 --arch gpu
${DIR}/build.sh --version 2.0.0 --arch cpu
${DIR}/build.sh --version 2.0.0 --arch gpu
