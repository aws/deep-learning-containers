#!/bin/bash
# Build the mooncake-transfer-engine-efa-cuda13 wheel from source on Rocky Linux 9
# (glibc 2.34, same as AL2023). PyPI has no aarch64 EFA wheel, and the aarch64 cuda13 wheel
# lacks the EFA and intra-node NVLink transports.
# Usage: MC_VERSION=<tag without v> OUTPUT_DIR=<dir> [MC_PATCH_DIR=<dir>] build_mooncake_efa.sh
set -euxo pipefail
MC_VERSION=${MC_VERSION:?}
OUTPUT_DIR=${OUTPUT_DIR:-/out}
MC_PATCH_DIR=${MC_PATCH_DIR:-}
CUDA_DASH=${CUDA_DASH:-13-0}
CUDA_ARCHS=${CUDA_ARCHS:-90;100;103}

dnf install -y epel-release dnf-plugins-core
dnf config-manager --set-enabled crb
dnf install -y git wget tar which findutils python3.12 python3.12-devel python3.12-pip
CUDA_REPO_ARCH=$(uname -m | sed s/aarch64/sbsa/)
dnf config-manager --add-repo "https://developer.download.nvidia.com/compute/cuda/repos/rhel9/${CUDA_REPO_ARCH}/cuda-rhel9.repo"
dnf install -y cuda-nvcc-${CUDA_DASH} cuda-cudart-devel-${CUDA_DASH} cuda-nvrtc-devel-${CUDA_DASH} \
  cuda-driver-devel-${CUDA_DASH} cuda-nvml-devel-${CUDA_DASH}
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
# Shared libraries link -lcuda; use the driver stub.
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}

# Same libfabric as the runtime image (/opt/amazon/efa); excluded from the wheel by auditwheel.
cd /tmp
curl -fsSL https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz | tar xz
(cd aws-efa-installer && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify)
test -f /opt/amazon/efa/include/rdma/fabric.h

git clone -q --depth 1 --branch "v${MC_VERSION}" https://github.com/kvcache-ai/Mooncake.git /src/mooncake
cd /src/mooncake
# Fail on a stale patch rather than silently building unpatched.
for p in ${MC_PATCH_DIR:+"$MC_PATCH_DIR"/*.patch}; do
  [ -f "$p" ] || continue
  echo "Applying $(basename "$p")"; git apply "$p"
done
bash dependencies.sh -y
export PATH=/usr/local/go/bin:$PATH
python3.12 -m pip install -q build wheel setuptools auditwheel patchelf

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/usr/bin/python3.12 \
  -DUSE_CUDA=ON -DUSE_EFA=ON -DUSE_INTRA_NVLINK=ON -DUSE_MNNVL=ON \
  -DUSE_HTTP=ON -DUSE_ETCD=ON -DSTORE_USE_ETCD=ON -DWITH_EP=OFF -DWITH_STORE_RUST=OFF \
  -DBUILD_UNIT_TESTS=OFF -DBUILD_BENCHMARK=OFF \
  -DLIBFABRIC_INCLUDE_DIR=/opt/amazon/efa/include -DLIBFABRIC_LIBRARY=/opt/amazon/efa/lib64/libfabric.so \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}"
make -j"$(nproc)"
make install

cd /src/mooncake
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:/opt/amazon/efa/lib64:${LD_LIBRARY_PATH:-}
EFA_CU13_BUILD=1 PYTHON_VERSION=3.12 OUTPUT_DIR="${OUTPUT_DIR}" ./scripts/build_wheel.sh
ls -la "${OUTPUT_DIR}"/*.whl
