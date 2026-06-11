# syntax=docker/dockerfile:1

# AWS-published vLLM DLC this image layers on top of. The build workflow passes
# the resolved base via --build-arg BASE_IMAGE (build_image.sh always forwards
# it); this default keeps standalone `docker build` working.
ARG BASE_IMAGE=public.ecr.aws/deep-learning-containers/vllm:0.22.1-gpu-py312-cu130-ubuntu22.04-sagemaker-v1.0

FROM ${BASE_IMAGE} AS base

LABEL maintainer="Hugging Face"
LABEL hf_dlc_major_version="1"

# Build-args the DLC harness (build_image.sh) always forwards. Declared so they
# are consumed (no "unconsumed build-arg" warnings); the SageMaker dlc.* labels
# themselves are attached by the harness, not here.
ARG FRAMEWORK
ARG FRAMEWORK_VERSION
ARG CONTAINER_TYPE

ARG TRANSFORMERS_VERSION=5.10.2
ARG HUGGINGFACE_HUB_VERSION=1.17.0
ARG HF_XET_VERSION=1.5.0

RUN apt-get update -y \
  && apt-get install -y --no-install-recommends curl unzip \
  && rm -rf /var/lib/apt/lists/*

RUN uv pip install --system \
  "transformers==${TRANSFORMERS_VERSION}" \
  "huggingface-hub==${HUGGINGFACE_HUB_VERSION}" \
  "hf-xet==${HF_XET_VERSION}" \
  "grpcio>=1.79.3" \
  && uv cache clean

# HF auto-optimization layer. Sourced by the entrypoint to set performance
# defaults at run time: expandable_segments allocator and LMCache CPU
# KV-offload. The bundled accelerators it enables (LMCache / NIXL /
# runai-streamer) already ship in the AWS base. See inference/vllm/README.md
# for the toggles.
COPY --chmod=755 ./scripts/huggingface/vllm/hf_optimizations.sh /usr/local/bin/hf_optimizations.sh


FROM base AS huggingface-vllm-sagemaker

# Daily cache-bust knob forwarded by build_image.sh (date stamp), matching the
# docker/vllm convention.
ARG CACHE_REFRESH=0

ENV HF_HUB_USER_AGENT_ORIGIN="aws:sagemaker:gpu-cuda:inference:hf-vllm"

# Copy CUDA compat and entrypoint scripts
COPY --chmod=755 ./scripts/huggingface/vllm/start_cuda_compat.sh /usr/local/bin/start_cuda_compat.sh
COPY --chmod=755 ./scripts/huggingface/vllm/sagemaker_entrypoint.sh /usr/local/bin/sagemaker_entrypoint.sh

# Fix ffmpeg related vulnerabilities CVE-2024-35366 / CVE-2024-35367 / CVE-2024-35368
RUN apt-get purge -y ffmpeg libav* || true && apt-get autoremove -y

ARG FFMPEG_TAG=n8.1

RUN apt-get update && apt-get install -y --no-install-recommends \
  pkg-config \
  nasm \
  yasm \
  libmp3lame-dev \
  libopus-dev \
  libvorbis-dev \
  libvpx-dev \
  libx264-dev \
  libx265-dev \
  && git clone https://git.ffmpeg.org/ffmpeg.git /tmp/ffmpeg \
  && cd /tmp/ffmpeg \
  && git checkout "${FFMPEG_TAG}" \
  && ./configure \
    --prefix=/usr/local \
    --disable-debug \
    --disable-doc \
    --enable-gpl \
    --enable-libmp3lame \
    --enable-libopus \
    --enable-libvorbis \
    --enable-libvpx \
    --enable-libx264 \
    --enable-libx265 \
  && make -j"$(nproc)" \
  && make install \
  && which ffmpeg \
  && ffmpeg -version \
  && ffprobe -version \
  && rm -rf /tmp/ffmpeg \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# NOTE: the mooncake-transfer-engine (CVE-2026-33186) and openssh
# (CVE-2026-35385/35414/35386) backports the v0.21 image carried are dropped
# here: the 0.22.1 base already ships the fixed versions
# (mooncake-transfer-engine 0.3.10.post2, openssh 1:8.9p1-3ubuntu0.15).
# ffmpeg above is still required (the base ships the vulnerable 4.4.2).

RUN HOME_DIR=/root \
  && uv pip install --system --upgrade pip requests PTable \
  && curl -o ${HOME_DIR}/oss_compliance.zip https://aws-dlinfra-utilities.s3.amazonaws.com/oss_compliance.zip \
  && unzip ${HOME_DIR}/oss_compliance.zip -d ${HOME_DIR}/ \
  && cp ${HOME_DIR}/oss_compliance/test/testOSSCompliance /usr/local/bin/testOSSCompliance \
  && chmod +x /usr/local/bin/testOSSCompliance \
  && chmod +x ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh \
  && ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh ${HOME_DIR} python3 \
  && rm -rf ${HOME_DIR}/oss_compliance*

ENTRYPOINT ["/usr/local/bin/sagemaker_entrypoint.sh"]