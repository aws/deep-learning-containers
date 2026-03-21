# ============================================================================
# PyTorch 2.10.0 DLC — Amazon Linux 2023
# Multi-stage: builder → base → eks
# ============================================================================

# ── Stage: builder ──────────────────────────────────────────────────────────
FROM nvidia/cuda:12.8.1-devel-amzn2023 AS builder

# System build deps
RUN dnf install -y --allowerasing \
  python3.12 python3.12-devel python3.12-pip \
  gcc gcc-c++ make cmake git openssl-devel ninja-build \
  tar xz curl \
  && dnf clean all

# Install cuDNN devel headers (needed for transformer-engine compilation)
RUN cd /tmp \
  && curl -sL https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.8.0.87_cuda12-archive.tar.xz | tar xJ \
  && cp -a cudnn-*/include/* /usr/local/cuda/include/ \
  && cp -a cudnn-*/lib/* /usr/local/cuda/lib64/ \
  && rm -rf /tmp/cudnn-* \
  && ldconfig

# uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Python venv
ENV UV_PROJECT_ENVIRONMENT="/opt/venv"
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Install Python deps from lockfile
COPY docker/pytorch/pyproject.toml docker/pytorch/uv.lock /tmp/build/
WORKDIR /tmp/build
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

# Install flash-attn and transformer-engine (need torch, no build isolation)
ARG FLASH_ATTN_VERSION=2.7.4.post1
ARG TRANSFORMER_ENGINE_VERSION=2.3.0
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --no-build-isolation \
  flash-attn==${FLASH_ATTN_VERSION} \
  transformer-engine[pytorch]==${TRANSFORMER_ENGINE_VERSION}
WORKDIR /

# GDRCopy userspace library
ARG GDRCOPY_VERSION=2.4.4
COPY scripts/pytorch/install_gdrcopy.sh /tmp/install_gdrcopy.sh
RUN bash /tmp/install_gdrcopy.sh ${GDRCOPY_VERSION} && rm /tmp/install_gdrcopy.sh


# ── Stage: base ─────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.8.1-runtime-amzn2023 AS base

LABEL maintainer="Amazon AI"
LABEL dlc_major_version="1"
LABEL framework="pytorch"
LABEL framework_version="2.10.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONIOENCODING=UTF-8 \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  DLC_CONTAINER_TYPE=training

# Minimal runtime deps (gcc needed for Triton/torch.compile JIT)
RUN dnf install -y --allowerasing \
  python3.12 shadow-utils tar gzip curl which findutils \
  util-linux cuda-nvcc-12-8 gcc gcc-c++ \
  && dnf clean all

# Python venv from builder
COPY --from=builder /opt/venv /opt/venv

# cuDNN runtime libs from builder (needed by transformer-engine)
COPY --from=builder /usr/local/cuda/lib64/libcudnn* /usr/local/cuda/lib64/

# GDRCopy userspace lib from builder
COPY --from=builder /usr/local/lib/libgdrapi.so* /usr/local/lib/
RUN ldconfig

# EFA + OpenMPI
ARG EFA_VERSION=1.47.0
COPY scripts/pytorch/install_efa.sh /tmp/install_efa.sh
RUN bash /tmp/install_efa.sh ${EFA_VERSION} && rm /tmp/install_efa.sh

# OpenSSH (root, port 22)
COPY scripts/pytorch/configure_ssh.sh /tmp/configure_ssh.sh
RUN bash /tmp/configure_ssh.sh && rm /tmp/configure_ssh.sh

# NCCL config
COPY scripts/pytorch/configure_nccl.sh /tmp/configure_nccl.sh
RUN bash /tmp/configure_nccl.sh && rm /tmp/configure_nccl.sh

# SageMaker BYOC paths
RUN mkdir -p /opt/ml/input/data /opt/ml/model /opt/ml/output /opt/ml/code

# Entrypoint
COPY scripts/pytorch/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# PATH and LD_LIBRARY_PATH
ENV PATH="/opt/venv/bin:/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:/usr/local/cuda/bin:${PATH}" \
  LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib:/opt/amazon/openmpi/lib:/opt/amazon/efa/lib:/usr/local/cuda/lib64:/usr/local/lib:${LD_LIBRARY_PATH}"

EXPOSE 22
WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]


# ── Stage: eks ──────────────────────────────────────────────────────────────
FROM base AS eks

# Non-root user
RUN groupadd -g 1000 mluser \
  && useradd -m -u 1000 -g 1000 mluser \
  && chown -R mluser:mluser /opt/venv /opt/ml /workspace

# SSH reconfigured for non-root (port 2222)
COPY scripts/pytorch/configure_ssh_eks.sh /tmp/configure_ssh_eks.sh
RUN bash /tmp/configure_ssh_eks.sh && rm /tmp/configure_ssh_eks.sh

EXPOSE 2222
USER mluser
WORKDIR /home/mluser