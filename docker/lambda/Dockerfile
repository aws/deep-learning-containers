ARG CUDA_VERSION=13.0.3
ARG PYTHON_VERSION=3.13
ARG DLC_MAJOR_VERSION=1
ARG DLC_MINOR_VERSION=0
ARG FFMPEG_VERSION=8.0.1
ARG NV_CODEC_HEADERS_VERSION=12.2.72.0
ARG AWSLAMBDARIC_VERSION=3.1.1
ARG FRAMEWORK=lambda
ARG FRAMEWORK_VERSION=13.0.3
ARG CONTAINER_TYPE=general

# ============================================================
# Lambda Python sources
# We take only the Python binary, stdlib, and lib-dynload.
# Site-packages are intentionally excluded — all dependencies
# (including awslambdaric) are installed via uv so we own the
# full dependency tree with no Lambda-bundled extras.
# ============================================================
FROM public.ecr.aws/lambda/python:${PYTHON_VERSION} as lambda-python

# ============================================================
# Lambda RIE (downloaded once, copied into runtime stages)
# ============================================================
FROM public.ecr.aws/lambda/python:${PYTHON_VERSION} as rie-downloader
RUN curl -Lo /usr/local/bin/aws-lambda-rie \
  https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie \
  && chmod +x /usr/local/bin/aws-lambda-rie

# ============================================================
# Shared builder base (CUDA runtime + uv)
# ============================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-amzn2023 as builder-base
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN dnf install -y --setopt=install_weak_deps=False tar gzip findutils \
  && dnf clean all && rm -rf /var/cache/dnf

# ============================================================
# Shared builder base — devel (CUDA devel + uv + git)
# ============================================================
FROM nvidia/cuda:${CUDA_VERSION}-devel-amzn2023 as builder-base-devel
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN dnf install -y --setopt=install_weak_deps=False tar gzip git findutils \
  && dnf clean all && rm -rf /var/cache/dnf

# ============================================================
# FFmpeg builder (compiled from source with NVENC/NVDEC GPU acceleration)
# ============================================================
FROM nvidia/cuda:${CUDA_VERSION}-devel-amzn2023 as ffmpeg-builder
ARG FFMPEG_VERSION
ARG NV_CODEC_HEADERS_VERSION
ENV PATH="/usr/local/cuda/bin:${PATH}"
RUN dnf install -y --setopt=install_weak_deps=False \
  tar xz gcc gcc-c++ make pkgconfig nasm yasm git diffutils \
  && dnf clean all && rm -rf /var/cache/dnf \
  && git clone --branch n${NV_CODEC_HEADERS_VERSION} --depth 1 https://github.com/FFmpeg/nv-codec-headers.git /tmp/nv-codec-headers \
  && cd /tmp/nv-codec-headers && make install && cd /tmp && rm -rf nv-codec-headers \
  && curl -fsSL https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n${FFMPEG_VERSION}.tar.gz | tar xz -C /tmp \
  && mv /tmp/FFmpeg-n${FFMPEG_VERSION} /tmp/ffmpeg \
  && cd /tmp/ffmpeg \
  && PKG_CONFIG_PATH="/usr/local/lib/pkgconfig" ./configure \
    --prefix=/usr/local \
    --disable-static \
    --enable-shared \
    --disable-doc \
    --disable-htmlpages \
    --disable-manpages \
    --disable-podpages \
    --disable-txtpages \
    --disable-debug \
    --enable-nonfree \
    --enable-cuda-nvcc \
    --extra-cflags=-I/usr/local/cuda/include \
    --extra-ldflags=-L/usr/local/cuda/lib64 \
    --nvccflags="-gencode arch=compute_75,code=sm_75 -O2" \
  && make -j$(nproc) \
  && make install \
  && strip /usr/local/bin/ffmpeg /usr/local/bin/ffprobe \
  && rm -rf /tmp/ffmpeg

# ============================================================
# Base builder (Python 3.13 — awslambdaric, requests, pip-licenses)
# ============================================================
FROM builder-base as builder-base-py3
ARG PYTHON_VERSION
COPY --from=lambda-python /var/lang/bin /var/lang/bin
COPY --from=lambda-python /var/lang/include /var/lang/include
COPY --from=lambda-python /var/lang/lib /var/lang/lib
RUN rm -rf /var/lang/lib/python${PYTHON_VERSION}/site-packages
ENV PATH="/var/lang/bin:$PATH" \
  UV_PROJECT_ENVIRONMENT="/var/lang"
RUN printf 'home = /var/lang/bin\nimplementation = CPython\n' >/var/lang/pyvenv.cfg
COPY ./docker/lambda/base/pyproject.toml ./docker/lambda/base/uv.lock /tmp/build/
WORKDIR /tmp/build
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-dev --no-install-project \
  && (find /var/lang -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true) \
  && find /var/lang -type f -name "*.pyc" -delete \
  && find /var/lang -type f -name "*.pyo" -delete

# ============================================================
# CuPy builder (Python 3.13 — awslambdaric, CuPy, NumPy, SciPy, etc.)
# ============================================================
FROM builder-base as builder-cupy-py3
ARG PYTHON_VERSION
COPY --from=lambda-python /var/lang/bin /var/lang/bin
COPY --from=lambda-python /var/lang/include /var/lang/include
COPY --from=lambda-python /var/lang/lib /var/lang/lib
RUN rm -rf /var/lang/lib/python${PYTHON_VERSION}/site-packages
ENV PATH="/var/lang/bin:$PATH" \
  UV_PROJECT_ENVIRONMENT="/var/lang"
RUN printf 'home = /var/lang/bin\nimplementation = CPython\n' >/var/lang/pyvenv.cfg
COPY ./docker/lambda/cupy/pyproject.toml ./docker/lambda/cupy/uv.lock /tmp/build/
WORKDIR /tmp/build
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-dev --no-install-project \
  && (find /var/lang -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true) \
  && find /var/lang -type f -name "*.pyc" -delete \
  && find /var/lang -type f -name "*.pyo" -delete

# ============================================================
# PyTorch builder (Python 3.13 — awslambdaric, PyTorch, SAM2, transformers)
# ============================================================
FROM builder-base-devel as builder-pytorch-py3
ARG PYTHON_VERSION
COPY --from=lambda-python /var/lang/bin /var/lang/bin
COPY --from=lambda-python /var/lang/include /var/lang/include
COPY --from=lambda-python /var/lang/lib /var/lang/lib
RUN rm -rf /var/lang/lib/python${PYTHON_VERSION}/site-packages
ENV PATH="/var/lang/bin:$PATH" \
  UV_PROJECT_ENVIRONMENT="/var/lang"
RUN printf 'home = /var/lang/bin\nimplementation = CPython\n' >/var/lang/pyvenv.cfg
COPY ./docker/lambda/pytorch/pyproject.toml ./docker/lambda/pytorch/uv.lock /tmp/build/
WORKDIR /tmp/build
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-dev --no-install-project \
  && (find /var/lang -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true) \
  && (find /var/lang -type d -name tests -exec rm -rf {} + 2>/dev/null; true) \
  && find /var/lang -type f -name "*.pyc" -delete \
  && find /var/lang -type f -name "*.pyo" -delete

# ============================================================
# sglang builder (Python 3.13 — sglang engine + kernels, from source)
# ============================================================
# Built on builder-pytorch-py3 (same base as builder-vllm-py3), which already
# carries torch 2.11.0+cu130 and the lambda RIC/OSS tooling (awslambdaric, boto3,
# pip, pip-licenses, pytest) + setuptools<81. So this stage only adds sglang's own
# build deps + the sglang engine; torch is NOT reinstalled (parity inherited).
FROM builder-pytorch-py3 as builder-sglang-py3
ARG PYTHON_VERSION
ARG CUDA_VERSION
ARG SGLANG_REF=bc8b3ab1f55c60951b586d84d4aec773a6f654df
ARG SGLANG_KERNEL_VERSION=0.4.4
# sglang engine version, from the config's metadata.framework_version (e.g.
# 0.5.14+dlc1) — same source the serving amzn2023 image uses.
ARG FRAMEWORK_VERSION=0.5.14+dlc1
# Build deps: gcc/g++ (present from builder-base-devel), openssl/pkgconfig/perl
# (outlines-core Rust), protobuf (sglang router), + Rust toolchain.
RUN dnf install -y --setopt=install_weak_deps=False \
  gcc gcc-c++ openssl-devel pkgconfig perl protobuf-compiler protobuf-devel \
  && dnf clean all && rm -rf /var/cache/dnf \
  && curl --proto '=https' --tlsv1.2 --retry 3 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
# Stamp the sglang package version (built from a shallow git clone, so setuptools_scm
# can't derive it). Matches the serving amzn2023 image's format exactly:
# ${FRAMEWORK_VERSION}.amzn2023.${SGLANG_REF} — e.g. 0.5.14+dlc1.amzn2023.<commit>
# (the '+dlc1' makes '.amzn2023.<commit>' a PEP 440 local-version segment).
ENV PATH="/usr/local/cuda/bin:/var/lang/bin:/root/.cargo/bin:$PATH" \
  UV_PYTHON="/var/lang/bin/python${PYTHON_VERSION}" \
  SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=${FRAMEWORK_VERSION}.amzn2023.${SGLANG_REF}
# Install sglang from source at the pinned commit (matches the serving amzn2023
# image). Non-editable: the runtime copies only /var/lang. torch 2.11.0+cu130 is
# already installed from the pytorch base, so sglang[all] resolves against it and
# we do NOT reinstall torch.
COPY ./docker/lambda/sglang/torch-constraints.txt /tmp/torch-constraints.txt
RUN git clone https://github.com/sgl-project/sglang.git /tmp/sglang \
  && git -C /tmp/sglang checkout ${SGLANG_REF}
RUN --mount=type=cache,target=/root/.cache/uv \
  CUDA_SHORT="$(echo ${CUDA_VERSION} | cut -d. -f1,2 | tr -d '.')" \
  && uv pip install --python "${UV_PYTHON}" cmake ninja scikit-build-core \
  # sglang + all extras (flashinfer, outlines, xgrammar, flash-attn-4 prerelease, ...).
  # torch/vision/audio are already installed (cu130) from the pytorch base; the
  # --constraint pins them so sglang[all] can't pull a different torch — so we do
  # NOT reinstall torch afterward.
  && uv pip install --python "${UV_PYTHON}" --prerelease=allow \
     --constraint /tmp/torch-constraints.txt \
     "/tmp/sglang/python[all]" \
  # prebuilt cu130 sgl_kernel; sm100 variant is binary-compatible down to sm_89 (A10G/L4/L40S).
  # Source-built (kernel index is not hash-stable) so it stays a config pin, not a lock.
  && uv pip install --python "${UV_PYTHON}" --no-deps --reinstall-package sglang-kernel \
     "sglang-kernel==${SGLANG_KERNEL_VERSION}" --index-url "https://docs.sglang.ai/whl/cu${CUDA_SHORT}/" \
  # bump sglang transitive deps with fixed CVEs (ECR scan): pillow, diffusers, jaraco.context
  && uv pip install --python "${UV_PYTHON}" \
     "pillow>=12.3.0" "diffusers>=0.38.0" "jaraco.context>=6.1.0" \
  && rm -rf /tmp/sglang \
  && (find /var/lang -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true) \
  && (find /var/lang -type d -name tests -exec rm -rf {} + 2>/dev/null; true) \
  && find /var/lang -type f -name "*.pyc" -delete \
  && find /var/lang -type f -name "*.pyo" -delete

# ============================================================
# vllm builder (Python 3.13 — vLLM engine built from source, on the lambda pytorch base)
# ============================================================
# Built on builder-pytorch-py3, which already carries torch 2.11.0+cu130 and the
# lambda RIC/OSS tooling (awslambdaric, boto3, pip, pip-licenses, pytest). vllm is
# built FROM SOURCE at a pinned commit (matching the serving amzn2023 image) so the
# compiled CUDA ops target the cu130 toolchain and the Lambda GPU fleet arches,
# rather than shipping vllm's cu129 PyPI wheel. Only the resulting wheel is installed
# into /var/lang; the build toolchain and sccache stay in this throwaway stage.
#
# The build runs directly in /var/lang: it already carries torch 2.11.0+cu130
# (vllm's setup.py imports torch at build time to read the CUDA version) and
# setuptools <81 (vllm's build breaks on 81; the torch-2.11 closure already
# constrains it below 81), so no separate build venv is needed.
FROM builder-pytorch-py3 as builder-vllm-py3
ARG PYTHON_VERSION
ARG CUDA_VERSION
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_REF=7b3d595eb197d714052ce296cc8b124f0dc8af31
# vLLM package version stamped onto the source-built wheel (setuptools_scm can't
# derive it from a shallow clone). Matches the serving amzn2023 image's ref.
ARG VLLM_VERSION=0.24.0
ARG FLASHINFER_VERSION=0.6.12
# Fleet arches: A10G(8.6) L4/L40S(8.9) + g7 Blackwell(12.0); T4/sm_75 is out of
# scope for engines. 8.0 kept for forward-compat with A100-class if ever added.
ARG TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 12.0"
# Cap build parallelism: vllm's CUDA compile is memory-hungry (MAX_JOBS=32 OOM-froze
# the 185GB box). sccache makes rebuilds cheap; tune via --build-arg.
ARG MAX_JOBS=8
ARG NVCC_THREADS=2
ARG USE_SCCACHE=true
# Wheel-cache control (see scripts/ci/build/lambda/pre_build.sh):
#   USE_PREBUILT_WHEEL=1 + a wheel present in prebuilt_wheels/ → skip the ~85-min
#   source compile and install the cached wheel. Default 0 (compile) so a plain
#   `docker build` with empty context dirs still works.
ARG USE_PREBUILT_WHEEL=0
ENV UV_PYTHON="/var/lang/bin/python${PYTHON_VERSION}" \
  PATH="/usr/local/cuda/bin:/var/lang/bin:$PATH"

# Build tooling: git (clone), gcc/g++ already present from builder-base-devel.
RUN dnf install -y --setopt=install_weak_deps=False git \
  && dnf clean all && rm -rf /var/cache/dnf

COPY ./docker/lambda/vllm/torch-constraints.txt /tmp/torch-constraints.txt
# Cache inputs from the build context. scripts/ci/build/lambda/pre_build.sh creates
# these dirs (mkdir -p) before every vLLM build and populates them from S3 on a
# cache hit; they are NOT committed to the repo (matching docker/vllm amzn2023).
#   prebuilt_wheels/ → a cached vLLM wheel (on cache hit; empty on miss)
#   sccache-cache/   → warm sccache compiler cache (speeds a cache-miss compile)
COPY ./docker/lambda/vllm/prebuilt_wheels/ /tmp/prebuilt_wheels/
COPY ./docker/lambda/vllm/sccache-cache/ /root/.cache/sccache/
# Clone vllm at the pinned ref. VLLM_REF is a commit SHA (not a branch/tag), so
# init + fetch the specific commit rather than `clone --branch` (which rejects SHAs).
# Needed even on a wheel-cache hit: the install step reads vllm's requirements/.
RUN git init /tmp/vllm \
  && cd /tmp/vllm \
  && git remote add origin ${VLLM_REPO} \
  && git fetch --depth 1 origin ${VLLM_REF} \
  && git checkout FETCH_HEAD

# Produce the vLLM wheel into /tmp/vllm-dist: use the cached wheel if one was
# fetched, otherwise build from source against /var/lang's cu130 torch. sccache
# writes to /root/.cache/sccache (a real dir COPY'd from context, so post_build can
# export + upload it). --py-limited-api=cp38 → the abi3 wheel runs on py3.13.
# The wheel + sccache are NOT deleted here — the vllm-wheel-export / vllm-sccache-export
# scratch stages read them out for the S3 cache on a miss.
RUN --mount=type=cache,target=/root/.cache/uv \
  mkdir -p /tmp/vllm-dist \
  && if [ "${USE_PREBUILT_WHEEL}" = "1" ] && ls /tmp/prebuilt_wheels/*.whl >/dev/null 2>&1; then \
       echo "✅ Using pre-built vLLM wheel from cache — skipping compilation" \
       && cp /tmp/prebuilt_wheels/*.whl /tmp/vllm-dist/; \
     else \
       echo "🔧 Building vLLM wheel from source..." \
       && CUDA_SHORT="$(echo ${CUDA_VERSION} | cut -d. -f1,2 | tr -d '.')" \
       && cd /tmp/vllm \
       # build tools into /var/lang (torch already present); unsafe-best-match so the
       # stale cmake 3.25 on the pytorch cu130 index doesn't mask PyPI's cmake>=3.26.1.
       && UV_INDEX_STRATEGY=unsafe-best-match uv pip install --python "${UV_PYTHON}" \
          cmake ninja setuptools-scm setuptools-rust \
       && export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" MAX_JOBS="${MAX_JOBS}" NVCC_THREADS="${NVCC_THREADS}" \
          VLLM_TARGET_DEVICE=cuda SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_VERSION}+cu${CUDA_SHORT}" \
       && if [ "${USE_SCCACHE}" = "true" ]; then \
            curl -fsSL -o /tmp/sccache.tar.gz "https://github.com/mozilla/sccache/releases/download/v0.8.1/sccache-v0.8.1-x86_64-unknown-linux-musl.tar.gz" \
            && tar -xzf /tmp/sccache.tar.gz -C /tmp && mv /tmp/sccache-v0.8.1-*/sccache /usr/bin/sccache && rm -rf /tmp/sccache* \
            && export SCCACHE_DIR=/root/.cache/sccache SCCACHE_IDLE_TIMEOUT=0 CMAKE_BUILD_TYPE=Release \
               CMAKE_C_COMPILER_LAUNCHER=sccache CMAKE_CXX_COMPILER_LAUNCHER=sccache CMAKE_CUDA_COMPILER_LAUNCHER=sccache \
            && sccache --show-stats; \
          fi \
       # --no-build-isolation: build in /var/lang so setup.py sees the installed torch.
       && "${UV_PYTHON}" setup.py bdist_wheel --dist-dir=/tmp/vllm-dist --py-limited-api=cp38 \
       && (command -v sccache >/dev/null && sccache --show-stats || true); \
     fi

# Install the built vllm wheel + deps into the runtime venv (/var/lang).
# The wheel is installed --no-deps because vllm's requirements/cuda.txt pins
# nvidia-cutlass-dsl[cu13]==4.5.2 → cuda-python 12.9 → cuda-bindings<12.10, which
# is IRRECONCILABLE with torch 2.11.0+cu130 (needs cuda-bindings>=13.0.3). That
# cutlass-dsl/quack/humming cluster is only for the FA4 cute-DSL attention path,
# which we don't use (flashinfer is the attention backend). So we install the real
# runtime deps (common.txt + the non-broken cuda.txt extras) and drop those three.
RUN --mount=type=cache,target=/root/.cache/uv \
  CUDA_SHORT="$(echo ${CUDA_VERSION} | cut -d. -f1,2 | tr -d '.')" \
  && VLLM_WHL="$(ls -t /tmp/vllm-dist/vllm-*.whl | head -1)" \
  # Filtered cuda.txt: drop the FA4 cute-DSL cluster (nvidia-cutlass-dsl/quack/humming
  # → cuda-python 12.9, broken on cu130; we use flashinfer for attention). Written
  # INSIDE requirements/ so its `-r common.txt` relative include resolves. torch/
  # vision/audio are already installed (cu130) from the base; the --constraint pins
  # them so this resolve can't downgrade torch (it dropped to 2.10.0+cu128 without it)
  # — so we do NOT reinstall torch.
  && grep -vE 'nvidia-cutlass-dsl|quack-kernels|humming-kernels' \
       /tmp/vllm/requirements/cuda.txt > /tmp/vllm/requirements/lambda-runtime.txt \
  && UV_INDEX_STRATEGY=unsafe-best-match uv pip install --python "${UV_PYTHON}" \
     --constraint /tmp/torch-constraints.txt \
     -r /tmp/vllm/requirements/lambda-runtime.txt \
     --extra-index-url "https://download.pytorch.org/whl/cu${CUDA_SHORT}" \
  && uv pip install --python "${UV_PYTHON}" --no-deps "${VLLM_WHL}" \
  && uv pip install --python "${UV_PYTHON}" \
     "flashinfer-python==${FLASHINFER_VERSION}" "flashinfer-cubin==${FLASHINFER_VERSION}" \
  # bump vllm transitive deps with fixed CVEs (ECR scan) to match the other images
  && uv pip install --python "${UV_PYTHON}" "pillow>=12.3.0" \
  # NOTE: do NOT delete /tmp/vllm-dist or /root/.cache/sccache — the
  # vllm-wheel-export / vllm-sccache-export scratch stages read them out of this
  # builder for the S3 cache. This is a throwaway builder (only /var/lang is copied
  # into the runtime image), so leaving them costs nothing in the final image.
  && rm -rf /tmp/vllm \
  && (find /var/lang -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true) \
  && (find /var/lang -type d -name tests -exec rm -rf {} + 2>/dev/null; true) \
  && find /var/lang -type f -name "*.pyc" -delete \
  && find /var/lang -type f -name "*.pyo" -delete

# ============================================================
# vLLM cache-export stages (scratch — for S3 wheel/sccache upload on a cache miss)
# Built only on demand by the build-image action via EXPORT_TARGETS:
#   docker buildx build --target vllm-wheel-export   --output type=local,dest=docker/lambda/vllm/prebuilt_wheels
#   docker buildx build --target vllm-sccache-export --output type=local,dest=docker/lambda/vllm/sccache-cache
# post_build.sh then uploads the exported wheel + sccache to S3.
# ============================================================
FROM scratch as vllm-wheel-export
COPY --from=builder-vllm-py3 /tmp/vllm-dist/ /

FROM scratch as vllm-sccache-export
COPY --from=builder-vllm-py3 /root/.cache/sccache/ /

# ============================================================
# OSS compliance builders (isolated — prevents dependency closure modification)
# ============================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-amzn2023 as oss-base-py3
COPY --from=builder-base-py3 /var/lang /var/lang
COPY ./scripts/docker/common/setup_oss_compliance.sh /tmp/setup_oss_compliance.sh
RUN PATH="/var/lang/bin:$PATH" bash /tmp/setup_oss_compliance.sh python3 \
  && touch /root/THIRD_PARTY_SOURCE_CODE_URLS

FROM nvidia/cuda:${CUDA_VERSION}-runtime-amzn2023 as oss-cupy-py3
COPY --from=builder-cupy-py3 /var/lang /var/lang
COPY ./scripts/docker/common/setup_oss_compliance.sh /tmp/setup_oss_compliance.sh
RUN PATH="/var/lang/bin:$PATH" bash /tmp/setup_oss_compliance.sh python3 \
  && touch /root/THIRD_PARTY_SOURCE_CODE_URLS

FROM nvidia/cuda:${CUDA_VERSION}-runtime-amzn2023 as oss-pytorch-py3
ARG FFMPEG_VERSION
COPY --from=builder-pytorch-py3 /var/lang /var/lang
COPY --from=lambda-python /lib64/libssl.so.3 /lib64/libcrypto.so.3 /lib64/
COPY ./scripts/docker/common/setup_oss_compliance.sh /tmp/setup_oss_compliance.sh
RUN PATH="/var/lang/bin:$PATH" bash /tmp/setup_oss_compliance.sh python3 \
  && printf '\n** FFmpeg; version n%s -- https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n%s.tar.gz\n' "${FFMPEG_VERSION}" "${FFMPEG_VERSION}" >> /root/BUILD_FROM_SOURCE_PACKAGES_LICENCES \
  && printf '\nFFmpeg n%s https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n%s.tar.gz\n' "${FFMPEG_VERSION}" "${FFMPEG_VERSION}" >> /root/THIRD_PARTY_SOURCE_CODE_URLS \
  && printf '\n** FFprobe; version n%s -- https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n%s.tar.gz\n' "${FFMPEG_VERSION}" "${FFMPEG_VERSION}" >> /root/BUILD_FROM_SOURCE_PACKAGES_LICENCES \
  && printf '\nFFprobe n%s https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n%s.tar.gz\n' "${FFMPEG_VERSION}" "${FFMPEG_VERSION}" >> /root/THIRD_PARTY_SOURCE_CODE_URLS

FROM nvidia/cuda:${CUDA_VERSION}-runtime-amzn2023 as oss-sglang-py3
COPY --from=builder-sglang-py3 /var/lang /var/lang
COPY ./scripts/docker/common/setup_oss_compliance.sh /tmp/setup_oss_compliance.sh
RUN PATH="/var/lang/bin:$PATH" bash /tmp/setup_oss_compliance.sh python3 \
  && touch /root/THIRD_PARTY_SOURCE_CODE_URLS

FROM nvidia/cuda:${CUDA_VERSION}-runtime-amzn2023 as oss-vllm-py3
COPY --from=builder-vllm-py3 /var/lang /var/lang
COPY ./scripts/docker/common/setup_oss_compliance.sh /tmp/setup_oss_compliance.sh
RUN PATH="/var/lang/bin:$PATH" bash /tmp/setup_oss_compliance.sh python3 \
  && touch /root/THIRD_PARTY_SOURCE_CODE_URLS

# ============================================================
# Runtime: Base Python 3.13 (CUDA + Python + RIC + RIE)
# ============================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-amzn2023 as base-py3
ARG PYTHON_VERSION
ARG DLC_MAJOR_VERSION
ARG DLC_MINOR_VERSION
LABEL maintainer="Amazon AI"
LABEL dlc_major_version="${DLC_MAJOR_VERSION}"
LABEL dlc_minor_version="${DLC_MINOR_VERSION}"
ARG CACHE_REFRESH=0
RUN dnf upgrade -y --security --releasever latest && dnf clean all && rm -rf /var/cache/dnf
COPY --from=builder-base-py3 /var/lang /var/lang
COPY --from=lambda-python /var/runtime /var/runtime
COPY --from=rie-downloader /usr/local/bin/aws-lambda-rie /usr/local/bin/aws-lambda-rie
COPY ./scripts/docker/lambda/lambda_entrypoint.sh /lambda_entrypoint.sh
RUN ln -sf /var/lang/bin/python${PYTHON_VERSION} /var/lang/bin/python3 \
  && ln -sf /var/lang/bin/python3 /var/lang/bin/python
ENV PATH="/var/lang/bin:$PATH" \
  LD_LIBRARY_PATH="/var/lang/lib:/lib64:/usr/lib64:/var/runtime:/var/runtime/lib:/var/task:/var/task/lib:/opt/lib:/usr/local/cuda/lib64:/x86_64-bottlerocket-linux-gnu/sys-root/usr/lib/nvidia" \
  PYTHONPATH="/var/runtime" \
  PYTHONDONTWRITEBYTECODE=1 \
  DLC_CONTAINER_TYPE=general \
  LAMBDA_TASK_ROOT=/var/task \
  LAMBDA_RUNTIME_DIR=/var/runtime \
  LANG=en_US.UTF-8 \
  TZ=:/etc/localtime
COPY ./scripts/docker/telemetry/deep_learning_container.py /usr/local/bin/deep_learning_container.py
COPY ./scripts/docker/telemetry/bash_telemetry.sh.template /tmp/bash_telemetry.sh.template
ARG FRAMEWORK
ARG FRAMEWORK_VERSION
ARG CONTAINER_TYPE
RUN chmod +x /usr/local/bin/deep_learning_container.py \
  && sed -e "s/{{FRAMEWORK}}/${FRAMEWORK}/g" \
    -e "s/{{FRAMEWORK_VERSION}}/${FRAMEWORK_VERSION}/g" \
    -e "s/{{CONTAINER_TYPE}}/${CONTAINER_TYPE}/g" \
    /tmp/bash_telemetry.sh.template >/usr/local/bin/bash_telemetry.sh \
  && chmod +x /usr/local/bin/bash_telemetry.sh \
  && rm /tmp/bash_telemetry.sh.template \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/etc/bash.bashrc \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/etc/bashrc \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/root/.bashrc \
  && rm -rf /var/lang/lib/python${PYTHON_VERSION}/ensurepip/_bundled
COPY --from=oss-base-py3 /root/THIRD_PARTY_SOURCE_CODE_URLS /root/THIRD_PARTY_SOURCE_CODE_URLS
COPY --from=oss-base-py3 /root/PYTHON_PACKAGES_LICENSES /root/PYTHON_PACKAGES_LICENSES
COPY --from=oss-base-py3 /root/LINUX_PACKAGES_LICENSES /root/LINUX_PACKAGES_LICENSES
COPY --from=oss-base-py3 /root/BUILD_FROM_SOURCE_PACKAGES_LICENCES /root/BUILD_FROM_SOURCE_PACKAGES_LICENCES
COPY --from=oss-base-py3 /usr/local/bin/testOSSCompliance /usr/local/bin/testOSSCompliance
WORKDIR /var/task
ENTRYPOINT ["/lambda_entrypoint.sh", "python", "-m", "awslambdaric"]
CMD ["handler.handler"]

# ============================================================
# Runtime: CuPy Python 3.13 (base + CuPy, NumPy, SciPy)
# ============================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-amzn2023 as cupy-py3
ARG PYTHON_VERSION
ARG DLC_MAJOR_VERSION
ARG DLC_MINOR_VERSION
LABEL maintainer="Amazon AI"
LABEL dlc_major_version="${DLC_MAJOR_VERSION}"
LABEL dlc_minor_version="${DLC_MINOR_VERSION}"
ARG CACHE_REFRESH=0
RUN dnf upgrade -y --security --releasever latest && dnf clean all && rm -rf /var/cache/dnf
COPY --from=builder-cupy-py3 /var/lang /var/lang
COPY --from=lambda-python /var/runtime /var/runtime
COPY --from=rie-downloader /usr/local/bin/aws-lambda-rie /usr/local/bin/aws-lambda-rie
COPY ./scripts/docker/lambda/lambda_entrypoint.sh /lambda_entrypoint.sh
RUN ln -sf /var/lang/bin/python${PYTHON_VERSION} /var/lang/bin/python3 \
  && ln -sf /var/lang/bin/python3 /var/lang/bin/python
ENV PATH="/var/lang/bin:$PATH" \
  LD_LIBRARY_PATH="/var/lang/lib:/lib64:/usr/lib64:/var/runtime:/var/runtime/lib:/var/task:/var/task/lib:/opt/lib:/usr/local/cuda/lib64:/x86_64-bottlerocket-linux-gnu/sys-root/usr/lib/nvidia" \
  PYTHONPATH="/var/runtime" \
  PYTHONDONTWRITEBYTECODE=1 \
  DLC_CONTAINER_TYPE=general \
  LAMBDA_TASK_ROOT=/var/task \
  LAMBDA_RUNTIME_DIR=/var/runtime \
  LANG=en_US.UTF-8 \
  TZ=:/etc/localtime
COPY ./scripts/docker/telemetry/deep_learning_container.py /usr/local/bin/deep_learning_container.py
COPY ./scripts/docker/telemetry/bash_telemetry.sh.template /tmp/bash_telemetry.sh.template
ARG FRAMEWORK

ARG FRAMEWORK_VERSION
ARG CONTAINER_TYPE
RUN chmod +x /usr/local/bin/deep_learning_container.py \
  && sed -e "s/{{FRAMEWORK}}/${FRAMEWORK}/g" \
    -e "s/{{FRAMEWORK_VERSION}}/${FRAMEWORK_VERSION}/g" \
    -e "s/{{CONTAINER_TYPE}}/${CONTAINER_TYPE}/g" \
    /tmp/bash_telemetry.sh.template >/usr/local/bin/bash_telemetry.sh \
  && chmod +x /usr/local/bin/bash_telemetry.sh \
  && rm /tmp/bash_telemetry.sh.template \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/etc/bash.bashrc \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/etc/bashrc \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/root/.bashrc \
  && rm -rf /var/lang/lib/python${PYTHON_VERSION}/ensurepip/_bundled
COPY --from=oss-cupy-py3 /root/THIRD_PARTY_SOURCE_CODE_URLS /root/THIRD_PARTY_SOURCE_CODE_URLS
COPY --from=oss-cupy-py3 /root/PYTHON_PACKAGES_LICENSES /root/PYTHON_PACKAGES_LICENSES
COPY --from=oss-cupy-py3 /root/LINUX_PACKAGES_LICENSES /root/LINUX_PACKAGES_LICENSES
COPY --from=oss-cupy-py3 /root/BUILD_FROM_SOURCE_PACKAGES_LICENCES /root/BUILD_FROM_SOURCE_PACKAGES_LICENCES
COPY --from=oss-cupy-py3 /usr/local/bin/testOSSCompliance /usr/local/bin/testOSSCompliance
WORKDIR /var/task
ENTRYPOINT ["/lambda_entrypoint.sh", "python", "-m", "awslambdaric"]
CMD ["handler.handler"]

# ============================================================
# Runtime: PyTorch Python 3.13 (base + PyTorch, FFmpeg, SAM2)
# ============================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-amzn2023 as pytorch-py3
ARG PYTHON_VERSION
ARG DLC_MAJOR_VERSION
ARG DLC_MINOR_VERSION
LABEL maintainer="Amazon AI"
LABEL dlc_major_version="${DLC_MAJOR_VERSION}"
LABEL dlc_minor_version="${DLC_MINOR_VERSION}"
ARG CACHE_REFRESH=0
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
RUN dnf upgrade -y --security --releasever latest \
  && dnf install -y --setopt=install_weak_deps=False \
    libxcb libX11 libXext libXfixes alsa-lib \
  && dnf clean all && rm -rf /var/cache/dnf
COPY --from=builder-pytorch-py3 /var/lang /var/lang
COPY --from=lambda-python /var/runtime /var/runtime
COPY --from=ffmpeg-builder /usr/local/bin/ffmpeg /usr/local/bin/ffprobe /usr/local/bin/
COPY --from=ffmpeg-builder /usr/local/lib/libav*.so* /usr/local/lib/libsw*.so* /usr/local/lib/libpostproc.so* /usr/local/lib/
COPY --from=rie-downloader /usr/local/bin/aws-lambda-rie /usr/local/bin/aws-lambda-rie
COPY ./scripts/docker/lambda/lambda_entrypoint.sh /lambda_entrypoint.sh
RUN ln -sf /var/lang/bin/python${PYTHON_VERSION} /var/lang/bin/python3 \
  && ln -sf /var/lang/bin/python3 /var/lang/bin/python
ENV PATH="/var/lang/bin:/usr/local/bin:$PATH" \
  LD_LIBRARY_PATH="/var/lang/lib:/lib64:/usr/lib64:/var/runtime:/var/runtime/lib:/var/task:/var/task/lib:/opt/lib:/usr/local/cuda/lib64:/usr/local/lib:/x86_64-bottlerocket-linux-gnu/sys-root/usr/lib/nvidia" \
  PYTHONPATH="/var/runtime" \
  PYTHONDONTWRITEBYTECODE=1 \
  TORCH_HOME=/tmp/torch \
  DLC_CONTAINER_TYPE=general \
  LAMBDA_TASK_ROOT=/var/task \
  LAMBDA_RUNTIME_DIR=/var/runtime \
  LANG=en_US.UTF-8 \
  TZ=:/etc/localtime
COPY ./scripts/docker/telemetry/deep_learning_container.py /usr/local/bin/deep_learning_container.py
COPY ./scripts/docker/telemetry/bash_telemetry.sh.template /tmp/bash_telemetry.sh.template
ARG FRAMEWORK
ARG FRAMEWORK_VERSION
ARG CONTAINER_TYPE
RUN chmod +x /usr/local/bin/deep_learning_container.py \
  && sed -e "s/{{FRAMEWORK}}/${FRAMEWORK}/g" \
    -e "s/{{FRAMEWORK_VERSION}}/${FRAMEWORK_VERSION}/g" \
    -e "s/{{CONTAINER_TYPE}}/${CONTAINER_TYPE}/g" \
    /tmp/bash_telemetry.sh.template >/usr/local/bin/bash_telemetry.sh \
  && chmod +x /usr/local/bin/bash_telemetry.sh \
  && rm /tmp/bash_telemetry.sh.template \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/etc/bash.bashrc \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/etc/bashrc \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/root/.bashrc \
  && rm -rf /var/lang/lib/python${PYTHON_VERSION}/ensurepip/_bundled
COPY --from=oss-pytorch-py3 /root/THIRD_PARTY_SOURCE_CODE_URLS /root/THIRD_PARTY_SOURCE_CODE_URLS
COPY --from=oss-pytorch-py3 /root/PYTHON_PACKAGES_LICENSES /root/PYTHON_PACKAGES_LICENSES
COPY --from=oss-pytorch-py3 /root/LINUX_PACKAGES_LICENSES /root/LINUX_PACKAGES_LICENSES
COPY --from=oss-pytorch-py3 /root/BUILD_FROM_SOURCE_PACKAGES_LICENCES /root/BUILD_FROM_SOURCE_PACKAGES_LICENCES
COPY --from=oss-pytorch-py3 /usr/local/bin/testOSSCompliance /usr/local/bin/testOSSCompliance
WORKDIR /var/task
ENTRYPOINT ["/lambda_entrypoint.sh", "python", "-m", "awslambdaric"]
CMD ["handler.handler"]

# ============================================================
# Runtime: sglang Python 3.13 (sglang serving engine)
# ============================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-amzn2023 as sglang-py3
ARG PYTHON_VERSION
ARG DLC_MAJOR_VERSION
ARG DLC_MINOR_VERSION
LABEL maintainer="Amazon AI"
LABEL dlc_major_version="${DLC_MAJOR_VERSION}"
LABEL dlc_minor_version="${DLC_MINOR_VERSION}"
ARG CACHE_REFRESH=0
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# sglang + flashinfer JIT-compile CUDA kernels at cold start, so the runtime needs
# a CUDA compiler toolchain AND math-library dev headers, not just libs: numactl-libs
# (libnuma.so.1 for sgl_kernel); gcc/g++/which (host compiler); cuda-nvcc/cudart-devel/crt
# (nvcc + CUDA headers); curand/cublas/cusolver/cusparse-devel (headers flashinfer's
# JIT norm/gemm kernels include, e.g. cublasLt.h). JIT caches write to /tmp (see ENV).
RUN dnf install -y --setopt=install_weak_deps=False \
  numactl-libs gcc gcc-c++ which \
  cuda-nvcc-13-0 cuda-cudart-devel-13-0 cuda-crt-13-0 \
  libcurand-devel-13-0 libcublas-devel-13-0 libcusolver-devel-13-0 libcusparse-devel-13-0 \
  && dnf upgrade -y --releasever=latest \
  && dnf clean all && rm -rf /var/cache/dnf
COPY --from=builder-sglang-py3 /var/lang /var/lang
COPY --from=lambda-python /var/runtime /var/runtime
COPY --from=rie-downloader /usr/local/bin/aws-lambda-rie /usr/local/bin/aws-lambda-rie
COPY ./scripts/docker/lambda/sglang_entrypoint.sh /sglang_entrypoint.sh
COPY ./scripts/docker/lambda/sglang/handler.py /var/task/handler.py
RUN ln -sf /var/lang/bin/python${PYTHON_VERSION} /var/lang/bin/python3 \
  && ln -sf /var/lang/bin/python3 /var/lang/bin/python
ENV PATH="/var/lang/bin:$PATH" \
  LD_LIBRARY_PATH="/var/lang/lib:/lib64:/usr/lib64:/var/runtime:/var/runtime/lib:/var/task:/var/task/lib:/opt/lib:/usr/local/cuda/lib64:/x86_64-bottlerocket-linux-gnu/sys-root/usr/lib/nvidia" \
  PYTHONPATH="/var/runtime" \
  PYTHONDONTWRITEBYTECODE=1 \
  TORCH_HOME=/tmp/torch \
  HF_HOME=/tmp/huggingface \
  TRITON_CACHE_DIR=/tmp/triton \
  FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer \
  TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
  USER=lambda \
  HOME=/tmp \
  XDG_CONFIG_HOME=/tmp/.config \
  XDG_CACHE_HOME=/tmp/.cache \
  CC=/usr/bin/gcc \
  CXX=/usr/bin/g++ \
  CUDA_HOME=/usr/local/cuda \
  DLC_CONTAINER_TYPE=general \
  LAMBDA_TASK_ROOT=/var/task \
  LAMBDA_RUNTIME_DIR=/var/runtime \
  LANG=en_US.UTF-8 \
  TZ=:/etc/localtime
# In multi-concurrency mode (AWS_LAMBDA_MAX_CONCURRENCY set) the RIC runs the
# handler without USER/LOGNAME/HOME; torch inductor's getpass.getuser() and libs
# writing to ~/.config would crash. USER + HOME=/tmp + XDG dirs + TORCHINDUCTOR
# cache in /tmp (the only writable FS on Lambda) prevent that.
COPY ./scripts/docker/telemetry/deep_learning_container.py /usr/local/bin/deep_learning_container.py
COPY ./scripts/docker/telemetry/bash_telemetry.sh.template /tmp/bash_telemetry.sh.template
ARG FRAMEWORK
ARG FRAMEWORK_VERSION
ARG CONTAINER_TYPE
RUN chmod +x /usr/local/bin/deep_learning_container.py \
  && sed -e "s/{{FRAMEWORK}}/${FRAMEWORK}/g" \
    -e "s/{{FRAMEWORK_VERSION}}/${FRAMEWORK_VERSION}/g" \
    -e "s/{{CONTAINER_TYPE}}/${CONTAINER_TYPE}/g" \
    /tmp/bash_telemetry.sh.template >/usr/local/bin/bash_telemetry.sh \
  && chmod +x /usr/local/bin/bash_telemetry.sh \
  && rm /tmp/bash_telemetry.sh.template \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/etc/bash.bashrc \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/etc/bashrc \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/root/.bashrc \
  && rm -rf /var/lang/lib/python${PYTHON_VERSION}/ensurepip/_bundled
COPY --from=oss-sglang-py3 /root/THIRD_PARTY_SOURCE_CODE_URLS /root/THIRD_PARTY_SOURCE_CODE_URLS
COPY --from=oss-sglang-py3 /root/PYTHON_PACKAGES_LICENSES /root/PYTHON_PACKAGES_LICENSES
COPY --from=oss-sglang-py3 /root/LINUX_PACKAGES_LICENSES /root/LINUX_PACKAGES_LICENSES
COPY --from=oss-sglang-py3 /root/BUILD_FROM_SOURCE_PACKAGES_LICENCES /root/BUILD_FROM_SOURCE_PACKAGES_LICENCES
COPY --from=oss-sglang-py3 /usr/local/bin/testOSSCompliance /usr/local/bin/testOSSCompliance
WORKDIR /var/task
ENTRYPOINT ["/sglang_entrypoint.sh", "python", "-m", "awslambdaric"]
CMD ["handler.handler"]

# ============================================================
# Runtime: vLLM Python 3.13 (lambda pytorch base + vLLM engine)
# ============================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-amzn2023 as vllm-py3
ARG PYTHON_VERSION
ARG DLC_MAJOR_VERSION
ARG DLC_MINOR_VERSION
LABEL maintainer="Amazon AI"
LABEL dlc_major_version="${DLC_MAJOR_VERSION}"
LABEL dlc_minor_version="${DLC_MINOR_VERSION}"
ARG CACHE_REFRESH=0
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# vllm + flashinfer JIT-compile CUDA kernels for the exact GPU arch at cold start,
# so the runtime needs a CUDA compiler toolchain AND math-library dev headers, not
# just libs: numactl-libs (libnuma.so.1); gcc/g++/which (host compiler);
# cuda-nvcc/cudart-devel/crt (nvcc + CUDA headers); curand/cublas/cusolver/cusparse
# -devel (headers flashinfer's sampling/gemm kernels include, e.g. curand.h). JIT
# caches write to /tmp (see ENV).
RUN dnf install -y --setopt=install_weak_deps=False \
  numactl-libs gcc gcc-c++ which \
  cuda-nvcc-13-0 cuda-cudart-devel-13-0 cuda-crt-13-0 \
  libcurand-devel-13-0 libcublas-devel-13-0 libcusolver-devel-13-0 libcusparse-devel-13-0 \
  && dnf upgrade -y --releasever=latest \
  && dnf clean all && rm -rf /var/cache/dnf
COPY --from=builder-vllm-py3 /var/lang /var/lang
COPY --from=lambda-python /var/runtime /var/runtime
COPY --from=rie-downloader /usr/local/bin/aws-lambda-rie /usr/local/bin/aws-lambda-rie
COPY ./scripts/docker/lambda/vllm_entrypoint.sh /vllm_entrypoint.sh
COPY ./scripts/docker/lambda/vllm/handler.py /var/task/handler.py
RUN chmod +x /vllm_entrypoint.sh \
  && ln -sf /var/lang/bin/python${PYTHON_VERSION} /var/lang/bin/python3 \
  && ln -sf /var/lang/bin/python3 /var/lang/bin/python
ENV PATH="/var/lang/bin:$PATH" \
  LD_LIBRARY_PATH="/var/lang/lib:/lib64:/usr/lib64:/var/runtime:/var/runtime/lib:/var/task:/var/task/lib:/opt/lib:/usr/local/cuda/lib64:/x86_64-bottlerocket-linux-gnu/sys-root/usr/lib/nvidia" \
  PYTHONPATH="/var/runtime" \
  PYTHONDONTWRITEBYTECODE=1 \
  TORCH_HOME=/tmp/torch \
  HF_HOME=/tmp/huggingface \
  TRITON_CACHE_DIR=/tmp/triton \
  FLASHINFER_WORKSPACE_BASE=/tmp/flashinfer \
  VLLM_CACHE_ROOT=/tmp/vllm \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
  USER=lambda \
  HOME=/tmp \
  XDG_CONFIG_HOME=/tmp/.config \
  XDG_CACHE_HOME=/tmp/.cache \
  CC=/usr/bin/gcc \
  CXX=/usr/bin/g++ \
  CUDA_HOME=/usr/local/cuda \
  DLC_CONTAINER_TYPE=general \
  LAMBDA_TASK_ROOT=/var/task \
  LAMBDA_RUNTIME_DIR=/var/runtime \
  LANG=en_US.UTF-8 \
  TZ=:/etc/localtime
# In multi-concurrency mode (AWS_LAMBDA_MAX_CONCURRENCY set), the RIC runs the
# handler without USER/LOGNAME set. torch inductor's cache_dir() calls
# getpass.getuser() when TORCHINDUCTOR_CACHE_DIR is unset → "OSError: No username
# set in the environment" crashes handler import. Setting TORCHINDUCTOR_CACHE_DIR
# (to /tmp, the only writable FS on Lambda) AND USER above prevents that.
COPY ./scripts/docker/telemetry/deep_learning_container.py /usr/local/bin/deep_learning_container.py
COPY ./scripts/docker/telemetry/bash_telemetry.sh.template /tmp/bash_telemetry.sh.template
ARG FRAMEWORK
ARG FRAMEWORK_VERSION
ARG CONTAINER_TYPE
RUN chmod +x /usr/local/bin/deep_learning_container.py \
  && sed -e "s/{{FRAMEWORK}}/${FRAMEWORK}/g" \
    -e "s/{{FRAMEWORK_VERSION}}/${FRAMEWORK_VERSION}/g" \
    -e "s/{{CONTAINER_TYPE}}/${CONTAINER_TYPE}/g" \
    /tmp/bash_telemetry.sh.template >/usr/local/bin/bash_telemetry.sh \
  && chmod +x /usr/local/bin/bash_telemetry.sh \
  && rm /tmp/bash_telemetry.sh.template \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/etc/bash.bashrc \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/etc/bashrc \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/root/.bashrc \
  && rm -rf /var/lang/lib/python${PYTHON_VERSION}/ensurepip/_bundled
COPY --from=oss-vllm-py3 /root/THIRD_PARTY_SOURCE_CODE_URLS /root/THIRD_PARTY_SOURCE_CODE_URLS
COPY --from=oss-vllm-py3 /root/PYTHON_PACKAGES_LICENSES /root/PYTHON_PACKAGES_LICENSES
COPY --from=oss-vllm-py3 /root/LINUX_PACKAGES_LICENSES /root/LINUX_PACKAGES_LICENSES
COPY --from=oss-vllm-py3 /root/BUILD_FROM_SOURCE_PACKAGES_LICENCES /root/BUILD_FROM_SOURCE_PACKAGES_LICENCES
COPY --from=oss-vllm-py3 /usr/local/bin/testOSSCompliance /usr/local/bin/testOSSCompliance
WORKDIR /var/task
ENTRYPOINT ["/vllm_entrypoint.sh", "python", "-m", "awslambdaric"]
CMD ["handler.handler"]

# ============================================================
# RIC builder (compile multi-mode concurrency RIC with C extensions)
# ============================================================
FROM builder-base as builder-ric-preview
ARG PYTHON_VERSION
ARG AWSLAMBDARIC_VERSION
RUN dnf install -y --setopt=install_weak_deps=False \
  gcc gcc-c++ cmake make autoconf automake libtool \
  && dnf clean all && rm -rf /var/cache/dnf
COPY --from=lambda-python /var/lang/bin /var/lang/bin
COPY --from=lambda-python /var/lang/include /var/lang/include
COPY --from=lambda-python /var/lang/lib /var/lang/lib
RUN rm -rf /var/lang/lib/python${PYTHON_VERSION}/site-packages
ENV PATH="/var/lang/bin:$PATH"
RUN printf 'home = /var/lang/bin\nimplementation = CPython\n' >/var/lang/pyvenv.cfg
COPY ./docker/lambda/artifacts/awslambdaric-${AWSLAMBDARIC_VERSION}.tar.gz /tmp/
RUN uv pip install --python /var/lang/bin/python${PYTHON_VERSION} --no-deps /tmp/awslambdaric-${AWSLAMBDARIC_VERSION}.tar.gz \
  && rm /tmp/awslambdaric-${AWSLAMBDARIC_VERSION}.tar.gz

# ============================================================
# Base Python 3.13 (+ multi-mode concurrency RIC)
# ============================================================
FROM base-py3 as base-preview-py3
ARG PYTHON_VERSION
RUN rm -rf /var/lang/lib/python${PYTHON_VERSION}/site-packages/awslambdaric* \
  && rm -f /var/lang/lib/python${PYTHON_VERSION}/site-packages/runtime_client*.so
COPY --from=builder-ric-preview /var/lang/lib/python${PYTHON_VERSION}/site-packages/ /var/lang/lib/python${PYTHON_VERSION}/site-packages/
# Default: process mode (RIC-native). Customers override per function at runtime
# to thread (one shared model copy in GPU VRAM) or hybrid; function-config env
# vars take precedence over this image default. In process/hybrid, VRAM scales
# with worker count (one model copy + CUDA context per process) — size function
# concurrency to the model's VRAM footprint.
ENV AWS_LAMBDA_CONCURRENCY_MODE=process

# ============================================================
# CuPy Python 3.13 (+ multi-mode concurrency RIC)
# ============================================================
FROM cupy-py3 as cupy-preview-py3
ARG PYTHON_VERSION
RUN rm -rf /var/lang/lib/python${PYTHON_VERSION}/site-packages/awslambdaric* \
  && rm -f /var/lang/lib/python${PYTHON_VERSION}/site-packages/runtime_client*.so
COPY --from=builder-ric-preview /var/lang/lib/python${PYTHON_VERSION}/site-packages/ /var/lang/lib/python${PYTHON_VERSION}/site-packages/
# Default: process mode (RIC-native). Customers override per function at runtime
# to thread (one shared model copy in GPU VRAM) or hybrid; function-config env
# vars take precedence over this image default. In process/hybrid, VRAM scales
# with worker count (one model copy + CUDA context per process) — size function
# concurrency to the model's VRAM footprint.
ENV AWS_LAMBDA_CONCURRENCY_MODE=process

# ============================================================
# PyTorch Python 3.13 (+ multi-mode concurrency RIC)
# ============================================================
FROM pytorch-py3 as pytorch-preview-py3
ARG PYTHON_VERSION
RUN rm -rf /var/lang/lib/python${PYTHON_VERSION}/site-packages/awslambdaric* \
  && rm -f /var/lang/lib/python${PYTHON_VERSION}/site-packages/runtime_client*.so
COPY --from=builder-ric-preview /var/lang/lib/python${PYTHON_VERSION}/site-packages/ /var/lang/lib/python${PYTHON_VERSION}/site-packages/
# Default: process mode (RIC-native). Customers override per function at runtime
# to thread (one shared model copy in GPU VRAM) or hybrid; function-config env
# vars take precedence over this image default. In process/hybrid, VRAM scales
# with worker count (one model copy + CUDA context per process) — size function
# concurrency to the model's VRAM footprint.
ENV AWS_LAMBDA_CONCURRENCY_MODE=process
# ============================================================
# sglang Python 3.13 (+ multi-mode concurrency RIC)
# ============================================================
FROM sglang-py3 as sglang-preview-py3
ARG PYTHON_VERSION
RUN rm -rf /var/lang/lib/python${PYTHON_VERSION}/site-packages/awslambdaric* \
  && rm -f /var/lang/lib/python${PYTHON_VERSION}/site-packages/runtime_client*.so
COPY --from=builder-ric-preview /var/lang/lib/python${PYTHON_VERSION}/site-packages/ /var/lang/lib/python${PYTHON_VERSION}/site-packages/
# Default: process mode (RIC-native). Customers override per function at runtime
# to thread (one shared model copy in GPU VRAM) or hybrid; function-config env
# vars take precedence over this image default. For serving engines, thread mode
# is recommended so concurrent invocations share one model copy in VRAM.
ENV AWS_LAMBDA_CONCURRENCY_MODE=process

# ============================================================
# vLLM Python 3.13 (+ multi-mode concurrency RIC)
# ============================================================
FROM vllm-py3 as vllm-preview-py3
ARG PYTHON_VERSION
RUN rm -rf /var/lang/lib/python${PYTHON_VERSION}/site-packages/awslambdaric* \
  && rm -f /var/lang/lib/python${PYTHON_VERSION}/site-packages/runtime_client*.so
COPY --from=builder-ric-preview /var/lang/lib/python${PYTHON_VERSION}/site-packages/ /var/lang/lib/python${PYTHON_VERSION}/site-packages/
# Default: process mode (RIC-native). Customers override per function at runtime
# to thread (one shared model copy in GPU VRAM) or hybrid; function-config env
# vars take precedence over this image default. For serving engines, thread mode
# is recommended so concurrent invocations share one model copy in VRAM.
ENV AWS_LAMBDA_CONCURRENCY_MODE=process