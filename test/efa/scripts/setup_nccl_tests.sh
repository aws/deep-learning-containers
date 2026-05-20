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

# Locate libnccl + headers. master's path (/usr/local) works on apt-based images;
# fall back to the nvidia-nccl-cu* wheel's site-packages dir for vLLM.
NCCL_HOME=/usr/local
if [ ! -f "${NCCL_HOME}/include/nccl.h" ]; then
    WHEEL_DIR=$(python3 -c "import nvidia.nccl, os; print(os.path.dirname(nvidia.nccl.__file__))" 2>/dev/null || true)
    if [ -n "${WHEEL_DIR}" ] && [ -f "${WHEEL_DIR}/include/nccl.h" ]; then
        NCCL_HOME="${WHEEL_DIR}"
        # Makefile links -lnccl, expects libnccl.so (no version suffix).
        if [ ! -e "${NCCL_HOME}/lib/libnccl.so" ]; then
            SONAME=$(basename "$(ls "${NCCL_HOME}"/lib/libnccl.so.* | head -1)")
            ln -s "${SONAME}" "${NCCL_HOME}/lib/libnccl.so"
        fi
        # mpirun children need to find libnccl.so.2 — wheel dir isn't on the
        # default loader path.
        echo "${NCCL_HOME}/lib" >/etc/ld.so.conf.d/nvidia-nccl.conf
        ldconfig
    else
        echo "ERROR: cannot locate nccl headers"
        exit 1
    fi
fi

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
