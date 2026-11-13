#!/usr/bin/env bash
# Build the TensorFlow Serving model server from source with hermetic CUDA,
# nvcc and an extended GPU target list (adds Blackwell sm_100 for B200/GB200
# and sm_120 for G7/G7e).
#
# Runs inside the tfs-build stage of docker/tensorflow/inference/<ver>/Dockerfile.cuda.
# TF Serving's .bazelrc only defines a clang CUDA config, so TensorFlow's nvcc
# config is appended (same lines as TensorFlow's .bazelrc).
#
# The output binary is verified to contain sm_100 and sm_120 code before it
# is written.
set -euo pipefail

TFS_REF=""
CUDA_VERSION=""
CUDNN_VERSION=""
NCCL_VERSION=""
NVSHMEM_VERSION=""
COMPUTE_CAPABILITIES=""
OUT_DIR=""
JOBS="$(nproc)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tfs-ref) TFS_REF="$2"; shift 2 ;;
    --cuda-version) CUDA_VERSION="$2"; shift 2 ;;
    --cudnn-version) CUDNN_VERSION="$2"; shift 2 ;;
    --nccl-version) NCCL_VERSION="$2"; shift 2 ;;
    --nvshmem-version) NVSHMEM_VERSION="$2"; shift 2 ;;
    --compute-capabilities) COMPUTE_CAPABILITIES="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done
for v in TFS_REF CUDA_VERSION CUDNN_VERSION NCCL_VERSION NVSHMEM_VERSION COMPUTE_CAPABILITIES OUT_DIR; do
  [[ -n "${!v}" ]] || { echo "ERROR: --${v,,} is required" >&2; exit 1; }
done

SRC=/opt/src/serving
rm -rf "${SRC}" && mkdir -p "${SRC}" && cd "${SRC}"
git init -q .
git remote add origin https://github.com/tensorflow/serving.git
git fetch -q --depth 1 origin "refs/tags/${TFS_REF}"
git checkout -q --detach FETCH_HEAD
echo "TensorFlow Serving ${TFS_REF} at $(git rev-parse HEAD)"

printf '%s\n' \
  'common:cuda_nvcc --config=cuda' \
  'common:cuda_nvcc --action_env=TF_NVCC_CLANG="1"' \
  'common:cuda_nvcc --@local_config_cuda//:cuda_compiler=nvcc' \
  >> .bazelrc

# Link against the CUDA driver stub (no GPU on the build host).
STUB_DIR=/usr/local/cuda/targets/x86_64-linux/lib/stubs
ln -sf "${STUB_DIR}/libcuda.so" "${STUB_DIR}/libcuda.so.1"

LD_LIBRARY_PATH="${STUB_DIR}:${LD_LIBRARY_PATH:-}" \
LIBRARY_PATH="${STUB_DIR}:${LIBRARY_PATH:-}" \
bazel build \
  --color=no --curses=no --nokeep_going \
  --show_progress_rate_limit=60 \
  --config=release --config=cuda_nvcc \
  --repo_env=HERMETIC_CUDA_VERSION="${CUDA_VERSION}" \
  --repo_env=HERMETIC_CUDNN_VERSION="${CUDNN_VERSION}" \
  --repo_env=HERMETIC_NCCL_VERSION="${NCCL_VERSION}" \
  --repo_env=HERMETIC_NVSHMEM_VERSION="${NVSHMEM_VERSION}" \
  --repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES="${COMPUTE_CAPABILITIES}" \
  --repo_env=TF_CUDA_COMPUTE_CAPABILITIES="${COMPUTE_CAPABILITIES}" \
  --jobs="${JOBS}" \
  --local_resources=memory=HOST_RAM*0.7 \
  --copt=-fPIC \
  --linkopt="-L${STUB_DIR}" \
  --action_env="LD_LIBRARY_PATH=${STUB_DIR}:${LD_LIBRARY_PATH:-}" \
  --action_env="LIBRARY_PATH=${STUB_DIR}:${LIBRARY_PATH:-}" \
  --verbose_failures \
  tensorflow_serving/model_servers:tensorflow_model_server
rm -f "${STUB_DIR}/libcuda.so.1"

mkdir -p "${OUT_DIR}"
cp bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server "${OUT_DIR}/"

# Fail the build if Blackwell code is missing.
ELF_ARCHS=$(cuobjdump --list-elf "${OUT_DIR}/tensorflow_model_server")
for arch in sm_100 sm_120; do
  [[ "${ELF_ARCHS}" == *"${arch}"* ]] || { echo "ERROR: tensorflow_model_server has no ${arch} cubins" >&2; exit 1; }
done
echo "Built ${OUT_DIR}/tensorflow_model_server (sm_100 + sm_120 verified)"
