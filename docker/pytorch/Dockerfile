# ============================================================================
# PyTorch DLC — Amazon Linux 2023
# Multi-stage: builder → runtime → runtime-eks
#
# All version defaults match versions.env. Workflows pass --build-arg after
# sourcing versions.env so there is a single source of truth.
# ============================================================================

# ── Global ARGs (available to all stages) ───────────────────────────────────
ARG CUDA_VERSION=12.8.1
ARG CUDNN_VERSION=9.8.0.87
ARG PYTHON_VERSION=3.12
ARG TORCH_VERSION=2.10.0
ARG DLC_MAJOR_VERSION=1
ARG DLC_MINOR_VERSION=0
ARG FLASH_ATTN_VERSION=2.7.4.post1
ARG TRANSFORMER_ENGINE_VERSION=2.3.0
ARG GDRCOPY_VERSION=2.4.4
ARG EFA_VERSION=1.47.0
# Compiler parallelism cap — prevents OOM on hosts where nproc > available RAM.
# Default 8 is safe for 32GB+; override with --build-arg MAX_JOBS=16 on larger hosts.
ARG MAX_JOBS=8

# ── Stage: builder ──────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-devel-amzn2023 AS builder
ARG PYTHON_VERSION
ARG CUDNN_VERSION
ARG FLASH_ATTN_VERSION
ARG TRANSFORMER_ENGINE_VERSION
ARG GDRCOPY_VERSION
ARG MAX_JOBS

# System build deps
RUN dnf install -y --allowerasing \
  python${PYTHON_VERSION} python${PYTHON_VERSION}-devel python${PYTHON_VERSION}-pip \
  gcc gcc-c++ make cmake git openssl-devel ninja-build \
  tar xz curl \
  && dnf clean all

# uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Python venv
ENV UV_PROJECT_ENVIRONMENT="/opt/venv"
RUN python${PYTHON_VERSION} -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Install Python deps from lockfile (changes only when pyproject.toml/uv.lock change)
COPY docker/pytorch/pyproject.toml docker/pytorch/uv.lock /tmp/build/
WORKDIR /tmp/build
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

# cuDNN devel headers — placed after uv sync so a cuDNN bump doesn't
# invalidate the (expensive) lockfile install layer above.
RUN cd /tmp \
  && curl -sL https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz | tar xJ \
  && cp -a cudnn-*/include/* /usr/local/cuda/include/ \
  && cp -a cudnn-*/lib/* /usr/local/cuda/lib64/ \
  && rm -rf /tmp/cudnn-* \
  && ldconfig

# flash-attn and transformer-engine — compiled together in a single layer.
# MAX_JOBS caps compiler parallelism to prevent OOM (each process ~2GB peak;
# default 8 is safe for 128GB builders running both compilations in parallel).
RUN --mount=type=cache,target=/root/.cache/uv MAX_JOBS=${MAX_JOBS} \
  uv pip install --no-build-isolation \
  flash-attn==${FLASH_ATTN_VERSION} \
  transformer-engine[pytorch]==${TRANSFORMER_ENGINE_VERSION}
WORKDIR /

# GDRCopy userspace library
COPY scripts/pytorch/install_gdrcopy.sh /tmp/install_gdrcopy.sh
RUN bash /tmp/install_gdrcopy.sh ${GDRCOPY_VERSION} && rm /tmp/install_gdrcopy.sh


# ── Stage: runtime (EC2 / ECS / Batch / SageMaker) ──────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-runtime-amzn2023 AS runtime
ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG TORCH_VERSION
ARG DLC_MAJOR_VERSION
ARG DLC_MINOR_VERSION
ARG EFA_VERSION

LABEL maintainer="Amazon AI"
LABEL dlc_major_version="${DLC_MAJOR_VERSION}"
LABEL dlc_minor_version="${DLC_MINOR_VERSION}"
LABEL framework="pytorch"
LABEL framework_version="${TORCH_VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONIOENCODING=UTF-8 \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  DLC_CONTAINER_TYPE=training

# Derive cuda-nvcc package name: 12.8.1 → cuda-nvcc-12-8
RUN CUDA_MAJOR_MINOR=$(echo "${CUDA_VERSION}" | cut -d. -f1,2 | tr '.' '-') \
  && dnf install -y --allowerasing \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-devel shadow-utils tar gzip curl which findutils \
    util-linux cuda-nvcc-${CUDA_MAJOR_MINOR} gcc gcc-c++ \
  && dnf clean all

# Python venv from builder
COPY --from=builder /opt/venv /opt/venv

# cuDNN runtime libs from builder (needed by transformer-engine)
COPY --from=builder /usr/local/cuda/lib64/libcudnn* /usr/local/cuda/lib64/

# GDRCopy userspace lib from builder
COPY --from=builder /usr/local/lib/libgdrapi.so* /usr/local/lib/
RUN ldconfig

# EFA + OpenMPI
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
  LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib64:/opt/amazon/openmpi/lib:/opt/amazon/efa/lib:/usr/local/cuda/lib64:/usr/local/lib:${LD_LIBRARY_PATH}"

EXPOSE 22
WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]


# ── Stage: runtime-eks ───────────────────────────────────────────────────────
FROM runtime AS runtime-eks

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