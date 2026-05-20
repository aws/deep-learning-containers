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

# Locate libnccl + headers. master's V1 path (/usr/local) works on apt-based
# DLC images; nvidia-nccl-cu* wheel ships at <site-packages>/nvidia/nccl/{include,lib}
# for vLLM (cu12 and cu13 wheels both use the same dir name).
locate_nccl_home() {
    if [ -f /usr/local/include/nccl.h ] && [ -e /usr/local/lib/libnccl.so ]; then
        echo /usr/local
        return 0
    fi
    local d
    for d in $(python3 -c "import sys; [print(p) for p in sys.path]" 2>/dev/null | grep -E 'site-packages|dist-packages'); do
        if [ -f "${d}/nvidia/nccl/include/nccl.h" ]; then
            echo "${d}/nvidia/nccl"
            return 0
        fi
    done
    return 1
}

NCCL_HOME=$(locate_nccl_home)
if [ -z "${NCCL_HOME}" ]; then
    echo "ERROR: cannot locate nccl headers; tried /usr/local and nvidia-nccl-cu* wheels" >&2
    echo "--- diagnostic: site-packages contents ---" >&2
    python3 -c "import sys; [print(p) for p in sys.path]" >&2 || true
    find / -name nccl.h 2>/dev/null | head -10 >&2 || true
    exit 1
fi
echo "Using NCCL_HOME=${NCCL_HOME}"

# Wheel layout: ensure libnccl.so symlink exists and the lib dir is on the loader
# path. Apt layout (/usr/local) already has both via ldconfig.
if [ "${NCCL_HOME}" != /usr/local ]; then
    if [ ! -e "${NCCL_HOME}/lib/libnccl.so" ]; then
        SONAME=$(basename "$(ls "${NCCL_HOME}"/lib/libnccl.so.* 2>/dev/null | head -1)")
        if [ -n "${SONAME}" ]; then
            ln -s "${SONAME}" "${NCCL_HOME}/lib/libnccl.so"
        fi
    fi
    echo "${NCCL_HOME}/lib" >/etc/ld.so.conf.d/nvidia-nccl.conf
    ldconfig
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
