#!/bin/bash
# Ensure /usr/local/bin/all_reduce_perf exists in the container.
#
# - PyTorch DLCs bake the binary at build time (Dockerfile.cuda) → no-op.
# - vLLM Ubuntu (FROM vllm/vllm-openai) ships NCCL only via the
#   nvidia-nccl-cu12 wheel and includes build-essential from install_efa.sh,
#   so we compile nccl-tests against the wheel here.
set -ex

if [ -x /usr/local/bin/all_reduce_perf ]; then
    echo "all_reduce_perf already present, skipping build"
    exit 0
fi

NCCL_HOME=$(python3 -c "import nvidia.nccl; print(nvidia.nccl.__path__[0])" 2>/dev/null || true)
if [ -z "${NCCL_HOME}" ] || [ ! -d "${NCCL_HOME}/include" ] || [ ! -d "${NCCL_HOME}/lib" ]; then
    echo "ERROR: nvidia-nccl wheel not found and binary not preinstalled"
    exit 1
fi

# nccl-tests' Makefile links -lnccl, expects libnccl.so (no version suffix).
if [ ! -e "${NCCL_HOME}/lib/libnccl.so" ]; then
    SONAME=$(basename "$(ls "${NCCL_HOME}"/lib/libnccl.so.* | head -1)")
    ln -s "${SONAME}" "${NCCL_HOME}/lib/libnccl.so"
fi

# Register wheel lib dir with ldconfig so mpirun-spawned children resolve
# libnccl.so.2. Upstream vLLM relies on PyTorch dlopen'ing it; raw
# all_reduce_perf processes don't get that.
echo "${NCCL_HOME}/lib" >/etc/ld.so.conf.d/nvidia-nccl.conf
ldconfig

SRC_DIR=$(mktemp -d)
trap 'rm -rf "${SRC_DIR}"' EXIT

cd "${SRC_DIR}"
curl -fsSL https://github.com/NVIDIA/nccl-tests/archive/refs/heads/master.tar.gz | tar xz
cd nccl-tests-master
make MPI=1 MPI_HOME=/opt/amazon/openmpi NCCL_HOME="${NCCL_HOME}" CUDA_HOME=/usr/local/cuda
cp build/all_reduce_perf /usr/local/bin/all_reduce_perf

/usr/local/bin/all_reduce_perf --help >/dev/null 2>&1 || \
    { ldd /usr/local/bin/all_reduce_perf; exit 1; }
echo "Built /usr/local/bin/all_reduce_perf"
