# OpenFold3 on DLC base image (AL2023 + CUDA 13.0), SageMaker async inference.
#
# Multi-stage build:
#   builder  — install conda env + OpenFold3 + model weights
#   runtime  — DLC devel base (kept: Triton/CUTLASS JIT compiles at warmup)

ARG BASE_IMAGE=public.ecr.aws/deep-learning-containers/base:devel-cu130-amzn2023

# =============================================================================
# builder
# =============================================================================
FROM ${BASE_IMAGE} AS builder

ARG MINIFORGE_VERSION=25.11.0-1
ARG OPENFOLD3_REF=0.4.1
ARG CHECKPOINT=of3-p2-155k.pt

# Build dependencies
RUN dnf install -y \
  wget \
  git \
  gcc gcc-c++ make \
  && dnf clean all

# Miniforge (conda)
RUN wget -q -P /tmp \
  "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-Linux-x86_64.sh" \
  && bash /tmp/Miniforge3-Linux-x86_64.sh -b -p /opt/conda \
  && rm /tmp/Miniforge3-Linux-x86_64.sh
ENV PATH=/opt/conda/bin:$PATH

# Clone OpenFold3 at a pinned ref (not HEAD)
WORKDIR /opt
RUN git clone --depth 1 --branch "${OPENFOLD3_REF}" https://github.com/aqlaboratory/openfold-3.git

# Install conda environment from upstream lock file
WORKDIR /opt/openfold-3
RUN mamba install -n base -c conda-forge conda-lock --yes \
  && conda-lock install --name openfold3 environments/production-linux-64.lock \
  && mamba clean --all --yes \
  && conda clean --all --yes

ENV PATH=/opt/conda/envs/openfold3/bin:$PATH \
  CONDA_PREFIX=/opt/conda/envs/openfold3 \
  CONDA_DEFAULT_ENV=openfold3

# Third party dependencies (CUTLASS for the DeepSpeed Evoformer attention kernel)
RUN /opt/openfold-3/scripts/install_third_party_dependencies.sh

# Install openfold3
RUN pip install --no-deps --editable .

# CVE remediation: upgrade vulnerable pure-Python utility libs (not part of the
# model dependency graph) in both the base and openfold3 conda envs. openssl,
# curl, and torch are upstream-pinned in OpenFold3's conda lock and are tracked
# via the ECR scan allowlist instead.
RUN /opt/conda/bin/pip install --upgrade "urllib3>=2.7.0" "dulwich>=1.2.5" \
  && pip install --upgrade \
    "GitPython>=3.1.50" "urllib3>=2.7.0" "pillow>=12.2.0" "msgpack>=1.2.1"

# Download model weights (public bucket, no auth)
ENV OPENFOLD_CACHE=/root/.openfold3
RUN mkdir -p /root/.openfold3 \
  && pip install awscli \
  && aws s3 cp "s3://openfold3-data/openfold3-parameters/${CHECKPOINT}" /root/.openfold3/ --no-sign-request

# Serving dependencies
RUN pip install flask gunicorn

# =============================================================================
# runtime (SageMaker)
# =============================================================================
FROM ${BASE_IMAGE} AS openfold3-sagemaker

ARG DLC_MAJOR_VERSION=1
ARG DLC_MINOR_VERSION=0

LABEL maintainer="Amazon AI"
LABEL dlc_major_version="${DLC_MAJOR_VERSION}"
LABEL dlc_minor_version="${DLC_MINOR_VERSION}"

# libgomp for OpenMP (may already be in base)
RUN dnf install -y --allowerasing libgomp && dnf clean all

COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /opt/openfold-3 /opt/openfold-3
COPY --from=builder /root/.openfold3 /root/.openfold3

ENV PATH=/opt/conda/envs/openfold3/bin:/opt/conda/bin:$PATH \
  CONDA_PREFIX=/opt/conda/envs/openfold3 \
  CONDA_DEFAULT_ENV=openfold3 \
  OPENFOLD_CACHE=/root/.openfold3 \
  CUTLASS_PATH=/opt/openfold-3/cutlass

# CUDA forward-compat shim (shared with other DLCs)
COPY ./scripts/docker/common/start_cuda_compat.sh /usr/local/bin/start_cuda_compat.sh
RUN chmod +x /usr/local/bin/start_cuda_compat.sh

# Serving code + entrypoint
COPY ./scripts/docker/openfold3/sagemaker_handler.py /opt/ml/code/sagemaker_handler.py
COPY ./scripts/docker/openfold3/sagemaker_serve.py /opt/ml/code/sagemaker_serve.py
COPY ./scripts/docker/openfold3/sagemaker_entrypoint.sh /usr/bin/serve
RUN chmod +x /usr/bin/serve

WORKDIR /opt/ml/code
ENV SAGEMAKER_PROGRAM=sagemaker_handler.py

ENTRYPOINT ["/usr/bin/serve"]