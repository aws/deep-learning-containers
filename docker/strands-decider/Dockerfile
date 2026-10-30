# strands-decider (https://github.com/strands-labs/strands-decider) on the AL2023 base DLC, for SageMaker inference.
# torch stays at 2.7.1, the version strands-decider's kernels and transformers pin are tested with.
#
#   builder    base devel: compiles causal-conv1d against the image's torch
#   sagemaker  base runtime: the locked environment, the compiled kernel, `strands-decider serve` on port 8080
#
# The model is not in the image: SageMaker copies it to /opt/ml/model before the container starts.

ARG BASE_DEVEL_IMAGE=public.ecr.aws/deep-learning-containers/base:devel-cu129-amzn2023
ARG BASE_RUNTIME_IMAGE=public.ecr.aws/deep-learning-containers/base:runtime-cu129-amzn2023

# =============================================================================
# builder
# =============================================================================
FROM ${BASE_DEVEL_IMAGE} AS builder

ARG CAUSAL_CONV1D_VERSION=1.7.0
# A100 (8.0), A10G (8.6), L4 and L40S (8.9), H100 (9.0)
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

COPY ./docker/strands-decider/pyproject.toml ./docker/strands-decider/uv.lock /tmp/deps/
RUN --mount=type=cache,target=/root/.cache/uv cd /tmp/deps \
  && uv sync --frozen --no-dev --no-install-project

# causal-conv1d is the fused kernel for the Gated DeltaNet conv layer (without it transformers runs a slower PyTorch
# path). Its upstream wheels are for Python 3.11 and torch builds this image doesn't have, so compile it.
RUN --mount=type=cache,target=/root/.cache/uv --mount=type=cache,target=/root/.cache/pip uv pip install --python /usr/local/bin/python3 ninja \
  && CAUSAL_CONV1D_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" MAX_JOBS="$(nproc)" \
    pip wheel --no-build-isolation --no-deps "causal-conv1d==${CAUSAL_CONV1D_VERSION}" -w /wheels

# =============================================================================
# sagemaker
# =============================================================================
FROM ${BASE_RUNTIME_IMAGE} AS sagemaker

ARG DLC_MAJOR_VERSION=1
ARG DLC_MINOR_VERSION=0

LABEL maintainer="Amazon AI"
LABEL dlc_major_version="${DLC_MAJOR_VERSION}"
LABEL dlc_minor_version="${DLC_MINOR_VERSION}"

# Triton compiles a small C launcher at run time.
ARG CACHE_REFRESH=0
RUN echo "cache refresh: ${CACHE_REFRESH}" \
  && dnf install -y --setopt=install_weak_deps=False gcc \
  && dnf upgrade -y --security --releasever latest \
  && dnf clean all && rm -rf /var/cache/dnf

COPY ./docker/strands-decider/pyproject.toml ./docker/strands-decider/uv.lock /tmp/deps/
RUN --mount=type=cache,target=/root/.cache/uv --mount=type=bind,from=builder,source=/wheels,target=/tmp/wheels cd /tmp/deps \
  && uv sync --frozen --no-dev --no-install-project \
  && uv pip install --python /usr/local/bin/python3 --no-deps /tmp/wheels/*.whl \
  && rm -rf /tmp/deps

COPY ./scripts/docker/common/setup_oss_compliance.sh /tmp/setup_oss_compliance.sh
RUN bash /tmp/setup_oss_compliance.sh python3 \
  && rm /tmp/setup_oss_compliance.sh \
  && rm -rf /root/oss_compliance* /tmp/tmp* /root/.cache

# DECIDER_CHECKPOINT is the model folder SageMaker copies to /opt/ml/model; DECIDER_ARGS adds
# `strands-decider serve` options, such as "--max-batch 8 --strict-window".
ENV DECIDER_CHECKPOINT=/opt/ml/model \
  DECIDER_DEVICE=cuda \
  DECIDER_ARGS="" \
  HF_HOME=/opt/hf \
  HF_HUB_OFFLINE=1

COPY ./scripts/docker/strands_decider/sagemaker_entrypoint.sh /usr/local/bin/sagemaker_entrypoint.sh
RUN chmod +x /usr/local/bin/sagemaker_entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/sagemaker_entrypoint.sh"]