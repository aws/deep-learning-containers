#!/usr/bin/env bash
# Build the TensorFlow GPU wheel from source with hermetic CUDA, nvcc and an
# extended GPU target list (adds Blackwell sm_100 for B200/GB200 and sm_120
# for G7/G7e).
#
# Runs inside the tf-wheel-build stage of docker/tensorflow/training/<ver>/Dockerfile.cuda.
# Uses TensorFlow's own release config (ci/official/wheel.sh) with two changes:
#   - --config=cuda_nvcc: the default clang path does not target sm_120
#   - HERMETIC_CUDA_COMPUTE_CAPABILITIES: upstream list + sm_100 + compute_120
# Config order matters: cuda_nvcc must come before cuda_wheel, otherwise the
# wheel rule rejects the build for bundling CUDA libraries.
#
# The checkout must match the pinned commit (--tf-commit), and the installed
# Bazel must match the tag's .bazelversion. As in upstream's
# ci/official/utilities/rename_and_verify_wheels.sh, the wheel is checked and
# tagged with auditwheel (manylinux_2_27). The GPU targets are verified by the
# Dockerfile on the installed library, on both cache-hit and fresh builds.
set -euo pipefail

TF_REF=""
TF_COMMIT=""
CUDA_VERSION=""
CUDNN_VERSION=""
COMPUTE_CAPABILITIES=""
PYTHON_VERSION=""
OUT_DIR=""
JOBS="$(nproc)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tf-ref) TF_REF="$2"; shift 2 ;;
    --tf-commit) TF_COMMIT="$2"; shift 2 ;;
    --cuda-version) CUDA_VERSION="$2"; shift 2 ;;
    --cudnn-version) CUDNN_VERSION="$2"; shift 2 ;;
    --compute-capabilities) COMPUTE_CAPABILITIES="$2"; shift 2 ;;
    --python-version) PYTHON_VERSION="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done
for v in TF_REF TF_COMMIT CUDA_VERSION CUDNN_VERSION COMPUTE_CAPABILITIES PYTHON_VERSION OUT_DIR; do
  [[ -n "${!v}" ]] || { echo "ERROR: --${v,,} is required" >&2; exit 1; }
done

SRC=/opt/src/tensorflow
rm -rf "${SRC}" && mkdir -p "${SRC}" && cd "${SRC}"
git init -q .
git remote add origin https://github.com/tensorflow/tensorflow.git
git fetch -q --depth 1 origin "refs/tags/${TF_REF}"
git checkout -q --detach FETCH_HEAD
HEAD_COMMIT=$(git rev-parse HEAD)
[[ "${HEAD_COMMIT}" == "${TF_COMMIT}" ]] \
  || { echo "ERROR: tag ${TF_REF} is at ${HEAD_COMMIT}, expected ${TF_COMMIT}" >&2; exit 1; }
echo "TensorFlow ${TF_REF} at ${HEAD_COMMIT}"

WANT_BAZEL=$(head -n1 .bazelversion | tr -d '[:space:]')
HAVE_BAZEL=$(bazel --version | awk '{print $2}')
[[ "${HAVE_BAZEL}" == "${WANT_BAZEL}" ]] \
  || { echo "ERROR: Bazel ${HAVE_BAZEL} installed, ${TF_REF} needs ${WANT_BAZEL} (.bazelversion)" >&2; exit 1; }

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

RAW_DIR=$(mktemp -d)
find bazel-bin/tensorflow/tools/pip_package -name 'tensorflow-*.whl' -exec cp {} "${RAW_DIR}/" \;
RAW_WHEELS=("${RAW_DIR}"/tensorflow-*.whl)
[[ ${#RAW_WHEELS[@]} -eq 1 && -f "${RAW_WHEELS[0]}" ]] \
  || { echo "ERROR: expected one wheel in bazel-bin, found ${#RAW_WHEELS[@]}" >&2; exit 1; }

# Same check and platform tag as upstream's release wheels (auditwheel repair
# --plat manylinux_2_27_x86_64). Fails if the wheel needs a newer glibc or an
# external library outside the manylinux policy.
REQS=$(mktemp)
cat > "${REQS}" <<'EOF'
auditwheel==6.8.2 --hash=sha256:05fd2dd4b743217cd5f66066adc95b03a079424b1f0a878ad1e40fe2cdea4704
packaging==26.3 --hash=sha256:d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c
pyelftools==0.33 --hash=sha256:f215ad5f47d3f1373a21496a6c9e0707c622840d0622f23ff7ce08678b020036
EOF
python3 -m pip install --no-cache-dir --require-hashes -r "${REQS}"
mkdir -p "${OUT_DIR}"
python3 -m auditwheel repair --plat manylinux_2_27_x86_64 --wheel-dir "${OUT_DIR}" "${RAW_WHEELS[0]}"
rm -rf "${RAW_DIR}" "${REQS}"

WHEELS=("${OUT_DIR}"/tensorflow-*.whl)
[[ ${#WHEELS[@]} -eq 1 && -f "${WHEELS[0]}" ]] \
  || { echo "ERROR: expected one wheel in ${OUT_DIR}, found ${#WHEELS[@]}" >&2; exit 1; }
echo "Built ${WHEELS[0]}"
