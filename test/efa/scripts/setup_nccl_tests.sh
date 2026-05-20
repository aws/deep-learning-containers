#!/bin/bash
# Build NVIDIA nccl-tests' all_reduce_perf at test time.
#
# Mirrors master's test/dlc_tests/container_tests/bin/efa/build_all_reduce_perf.sh:
# the binary is NOT baked into images (PyTorch DLC included) — it's compiled
# inside the test container on every run. Works because the test container
# runs on a p4d.24xlarge (~1TB RAM) which absorbs nccl-tests' template-heavy
# verifiable.cu compile.
#
# Behavior:
# - PyTorch DLC: NCCL_HOME=/usr/local works because libnccl-dev was apt-installed.
# - vLLM Ubuntu: nvidia-nccl-cu12 wheel ships headers + libs in site-packages;
#   the install_efa.sh path leaves build-essential in place.
set -ex

if [ -x /usr/local/bin/all_reduce_perf ]; then
    echo "all_reduce_perf already present at /usr/local/bin/all_reduce_perf"
    exit 0
fi

# Locate libnccl headers. master's V1 builds NCCL from source into /usr/local
# (heavy, but produces well-behaved headers). Building against the
# nvidia-nccl-cu13 wheel headers OOMed nvcc on verifiable.cu — the wheel ships
# headers from a different NCCL build with much heavier template expansion.
#
# Cheapest reliable workaround: install libnccl-dev from the cuda apt repo at
# test time. Same NCCL version the image runs against, "thin" headers like
# master's, ~50MB transient install, removed at end of test by container
# teardown.
NCCL_HOME=/usr
if [ ! -f "${NCCL_HOME}/include/nccl.h" ]; then
    echo "Installing libnccl-dev from apt..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y --no-install-recommends libnccl-dev
fi

if [ ! -f "${NCCL_HOME}/include/nccl.h" ]; then
    echo "ERROR: nccl.h still not present after apt install" >&2
    apt list --installed 2>/dev/null | grep -i nccl >&2 || true
    exit 1
fi
echo "Using NCCL_HOME=${NCCL_HOME}"

cd /tmp
rm -rf nccl-tests
git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1 MPI_HOME=/opt/amazon/openmpi NCCL_HOME="${NCCL_HOME}" CUDA_HOME=/usr/local/cuda
cp build/all_reduce_perf /usr/local/bin/all_reduce_perf
cd /tmp
rm -rf nccl-tests

/usr/local/bin/all_reduce_perf --help >/dev/null 2>&1 || \
    { ldd /usr/local/bin/all_reduce_perf; exit 1; }
echo "Built /usr/local/bin/all_reduce_perf"
