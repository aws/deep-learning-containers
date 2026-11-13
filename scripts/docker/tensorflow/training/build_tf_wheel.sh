#!/usr/bin/env bash
# Build the TensorFlow GPU wheel from source with hermetic CUDA, nvcc and an
# extended GPU target list (adds Blackwell sm_120 for G7/G7e).
#
# Runs inside the tf-wheel-build stage of docker/tensorflow/training/<ver>/Dockerfile.cuda.
# Uses TensorFlow's own release config (ci/official/wheel.sh) with two changes:
#   - --config=cuda_nvcc: the default clang path does not target sm_120
#   - HERMETIC_CUDA_COMPUTE_CAPABILITIES: upstream list + compute_120
# Config order matters: cuda_nvcc must come before cuda_wheel, otherwise the
# wheel rule rejects the build for bundling CUDA libraries.
#
# The output wheel is verified to contain sm_120 code before it is written.
set -euo pipefail

TF_REF=""
CUDA_VERSION=""
CUDNN_VERSION=""
COMPUTE_CAPABILITIES=""
PYTHON_VERSION=""
OUT_DIR=""
JOBS="$(nproc)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tf-ref) TF_REF="$2"; shift 2 ;;
    --cuda-version) CUDA_VERSION="$2"; shift 2 ;;
    --cudnn-version) CUDNN_VERSION="$2"; shift 2 ;;
    --compute-capabilities) COMPUTE_CAPABILITIES="$2"; shift 2 ;;
    --python-version) PYTHON_VERSION="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done
for v in TF_REF CUDA_VERSION CUDNN_VERSION COMPUTE_CAPABILITIES PYTHON_VERSION OUT_DIR; do
  [[ -n "${!v}" ]] || { echo "ERROR: --${v,,} is required" >&2; exit 1; }
done

SRC=/opt/src/tensorflow
rm -rf "${SRC}" && mkdir -p "${SRC}" && cd "${SRC}"
git init -q .
git remote add origin https://github.com/tensorflow/tensorflow.git
git fetch -q --depth 1 origin "refs/tags/${TF_REF}"
git checkout -q --detach FETCH_HEAD
echo "TensorFlow ${TF_REF} at $(git rev-parse HEAD)"

# The TF mirror returns 404 for the pinned LLVM archive and GitHub serves it
# with either of two checksums. Accept both, as TF Serving 2.21's WORKSPACE
# does: download without a fixed checksum, then fail unless it is a known one.
python3 - <<'PY'
p = "third_party/xla/third_party/repo.bzl"
pinned = "3f986184ee126677dbd77edb16d6b82c057ec869fefd7a9871979941e52e837a"
alternate = "00b1077e029fa57e6f2d9ac24936a49acf23ebc051b04f487131116258be6248"
old = """ctx.download_and_extract(
            url = ctx.attr.urls,
            sha256 = ctx.attr.sha256,
            type = ctx.attr.type,
            stripPrefix = ctx.attr.strip_prefix,
        )"""
new = f"""if ctx.attr.sha256 == "{pinned}":
            res = ctx.download(url = ctx.attr.urls, output = "llvm.tar.gz")
            if res.sha256 not in ["{pinned}", "{alternate}"]:
                fail("Unexpected LLVM archive sha256: " + res.sha256)
            ctx.extract(archive = "llvm.tar.gz", stripPrefix = ctx.attr.strip_prefix)
            ctx.delete("llvm.tar.gz")
        else:
            """ + old
s = open(p).read()
assert s.count(old) == 1, f"{p}: download_and_extract block not found"
open(p, "w").write(s.replace(old, new))
PY

bazel build \
  --color=no --curses=no --nokeep_going \
  --show_progress_rate_limit=60 \
  --config=release_gpu_linux --config=cuda_nvcc --config=cuda_wheel \
  --repo_env=HERMETIC_PYTHON_VERSION="${PYTHON_VERSION}" \
  --repo_env=USE_PYWRAP_RULES=True \
  --repo_env=HERMETIC_CUDA_VERSION="${CUDA_VERSION}" \
  --repo_env=HERMETIC_CUDNN_VERSION="${CUDNN_VERSION}" \
  --repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES="${COMPUTE_CAPABILITIES}" \
  --jobs="${JOBS}" \
  --local_resources=memory=HOST_RAM*0.7 \
  --verbose_failures \
  //tensorflow/tools/pip_package:wheel

mkdir -p "${OUT_DIR}"
find bazel-bin/tensorflow/tools/pip_package -name 'tensorflow-*.whl' -exec cp {} "${OUT_DIR}/" \;
WHEEL=$(ls "${OUT_DIR}"/tensorflow-*.whl)

# Fail the build if sm_120 code is missing.
CHECK_DIR=$(mktemp -d)
(cd "${CHECK_DIR}" && unzip -q "${WHEEL}" 'tensorflow/libtensorflow_cc.so.2')
ELF_ARCHS=$(cuobjdump --list-elf "${CHECK_DIR}/tensorflow/libtensorflow_cc.so.2")
[[ "${ELF_ARCHS}" == *sm_120* ]] || { echo "ERROR: ${WHEEL} has no sm_120 cubins" >&2; exit 1; }
rm -rf "${CHECK_DIR}"
echo "Built ${WHEEL} (sm_120 verified)"
