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
# Cheapest reliable workaround: install the libnccl headers package at test
# time. Same NCCL version the image runs against, "thin" headers like master's,
# ~50MB transient install, removed at end of test by container teardown.
#
# Package manager differs by base image:
# - Ubuntu (PyTorch DLC, vLLM): apt's libnccl-dev from the cuda apt repo.
# - AL2023 (SGLang): dnf's libnccl-devel from the cuda-rhel9 repo. The image
#   installs libnccl + libnccl-devel at build time then removes libnccl-devel,
#   leaving the cuda-rhel9 repo configured so we can reinstall the headers here.
NCCL_HOME=/usr
if [ ! -f "${NCCL_HOME}/include/nccl.h" ]; then
    if command -v apt-get >/dev/null 2>&1; then
        echo "Installing libnccl-dev from apt..."
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq
        apt-get install -y --no-install-recommends libnccl-dev
    elif command -v dnf >/dev/null 2>&1; then
        echo "Installing libnccl-devel from dnf..."
        dnf install -y --setopt=install_weak_deps=False libnccl-devel
    elif command -v yum >/dev/null 2>&1; then
        echo "Installing libnccl-devel from yum..."
        yum install -y libnccl-devel
    else
        echo "ERROR: no supported package manager (apt-get/dnf/yum) found" >&2
        exit 1
    fi
fi

if [ ! -f "${NCCL_HOME}/include/nccl.h" ]; then
    echo "ERROR: nccl.h still not present after package install" >&2
    { apt list --installed 2>/dev/null || rpm -qa 2>/dev/null; } | grep -i nccl >&2 || true
    exit 1
fi
echo "Using NCCL_HOME=${NCCL_HOME}"

# Pre-build diagnostics so failures have actionable evidence.
echo "=== pre-build state ==="
free -h || true
df -h / /tmp || true
nproc || true
cat /sys/fs/cgroup/memory.max 2>/dev/null || \
    cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || true
echo "======================="

# On failure (including SIGKILL), dump everything we can to surface the cause.
trap '
RC=$?
echo "=== post-build (RC=$RC) ===" >&2
free -h >&2 || true
df -h / /tmp >&2 || true
ls -la /tmp/nccl-tests/build/*.o /tmp/nccl-tests/build/verifiable/ 2>/dev/null >&2 || true
cat /sys/fs/cgroup/memory.peak 2>/dev/null >&2 || \
    cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes 2>/dev/null >&2 || true
echo "==========================" >&2
exit $RC
' EXIT

# Single-arch + serial build: avoids any cgroup memory headroom issues from
# parallel nvcc/cicc forks. p4d hosts default to compute_80; build for that
# only — nccl-tests default targets ~9 archs, peak memory >2× higher.
SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
[ -z "${SM}" ] && SM=80  # safe default for p4d
NVCC_GENCODE="-gencode=arch=compute_${SM},code=sm_${SM}"

cd /tmp
rm -rf nccl-tests
git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make -j1 MPI=1 MPI_HOME=/opt/amazon/openmpi NCCL_HOME="${NCCL_HOME}" \
    CUDA_HOME=/usr/local/cuda NVCC_GENCODE="${NVCC_GENCODE}"
cp build/all_reduce_perf /usr/local/bin/all_reduce_perf
cd /tmp
rm -rf nccl-tests

/usr/local/bin/all_reduce_perf --help >/dev/null 2>&1 || \
    { ldd /usr/local/bin/all_reduce_perf; exit 1; }
echo "Built /usr/local/bin/all_reduce_perf"
