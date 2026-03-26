# ============================================================================
# XGBoost DLC — Ubuntu 22.04 (CUDA 12.6)
# Multi-stage build:
#
#   sqlite-builder ─── (compiles SQLite from source)
#   wheel-builder ──── (builds sagemaker_xgboost_container wheel)
#   base ──────────── (miniforge, XGBoost, system deps, telemetry)
#   xgboost-sagemaker  (SageMaker serving layer on top of base)
#
# Build targets:
#   --target base              → base image only
#   --target xgboost-sagemaker → full SageMaker image (default)
#
# Works with both:
#   - GitHub Actions (DLC main branch PR workflow)
#   - CodePipeline / CodeBuild (existing buildspec)
# ============================================================================

# ── Global ARGs ─────────────────────────────────────────────────────────────
ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=12.6.3
ARG MINIFORGE_VERSION=26.1.1-3
ARG MINIFORGE_CHECKSUM=b25b828b702df4dd2a6d24d4eb56cfa912471dd8e3342cde2c3d86fe3dc2d870
ARG PYTHON_VERSION=3.11
ARG PYARROW_VERSION=22.0.0
ARG XGBOOST_VERSION=3.0.5
ARG SAGEMAKER_XGBOOST_VERSION=3.0-5

# ── Stage: sqlite-builder ───────────────────────────────────────────────────
FROM ubuntu:${UBUNTU_VERSION} AS sqlite-builder
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential wget ca-certificates \
  && cd /tmp \
  && wget https://www.sqlite.org/2025/sqlite-autoconf-3500200.tar.gz \
  && tar xzf sqlite-autoconf-3500200.tar.gz \
  && cd sqlite-autoconf-3500200 \
  && ./configure --prefix=/usr/local && make && make install && ldconfig \
  && cd / && rm -rf /tmp/sqlite-autoconf-* \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Stage: wheel-builder ───────────────────────────────────────────────────
# Builds the sagemaker_xgboost_container wheel inside Docker so the build is
# self-contained and does not depend on host-side `python3 setup.py bdist_wheel`.
# Clones the source from GitHub since this DLC repo does not contain the
# sagemaker-xgboost-container package source.
FROM ubuntu:${UBUNTU_VERSION} AS wheel-builder
RUN apt-get update && apt-get install -y --no-install-recommends \
  python3 python3-pip python3-setuptools python3-wheel git ca-certificates \
  && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/aws/sagemaker-xgboost-container.git /build
WORKDIR /build
RUN python3 setup.py bdist_wheel --universal

# ── Stage: base ─────────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION} AS base

ARG MINIFORGE_VERSION
ARG MINIFORGE_CHECKSUM
ARG PYTHON_VERSION
ARG PYARROW_VERSION
ARG XGBOOST_VERSION

LABEL maintainer="Amazon AI"
LABEL dlc_major_version="1"

ENV DEBIAN_FRONTEND=noninteractive \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONIOENCODING=utf-8

# System dependencies
RUN apt-key del 7fa2af80 \
  && apt-get update && apt-get install -y --no-install-recommends wget \
  && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
  && dpkg -i cuda-keyring_1.1-1_all.deb \
  && apt-get update && apt-get -y upgrade \
  && apt-get -y install --no-install-recommends \
    build-essential curl git jq libatlas-base-dev expat nginx \
    openjdk-11-jdk-headless unzip wget apparmor linux-libc-dev \
    libxml2 libgstreamer1.0-0 libglib2.0-0 \
    apt-transport-https ca-certificates gnupg software-properties-common \
    libcurl4-openssl-dev libssl-dev \
    python3-dev zlib1g-dev zstd libsqlite3-0 \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Miniforge (conda-forge by default, no Anaconda license required)
RUN curl -L --output /tmp/Miniforge3.sh \
  https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh \
  && echo "${MINIFORGE_CHECKSUM} /tmp/Miniforge3.sh" | sha256sum -c - \
  && bash /tmp/Miniforge3.sh -bfp /miniforge3 \
  && rm /tmp/Miniforge3.sh

ENV PATH=/miniforge3/bin:${PATH}

# Conda packages
RUN conda config --system --set channel_priority strict \
  && conda config --system --set auto_update_conda false \
  && conda config --system --set show_channel_urls true \
  && conda install -y \
    python=${PYTHON_VERSION} \
    requests=2.32.3 \
    "openssl>=3.6.1" \
    pyarrow=${PYARROW_VERSION} \
  && conda clean -afy

# MLIO (awslabs/ml-io) is no longer built from source — the project is
# unmaintained (last release 2020) and its C++ dependencies fail to compile
# with modern CMake/GCC on Ubuntu 22.04. A stub module is installed so that
# imports succeed but raises a clear error if pipe mode is used at runtime.
COPY docker/xgboost/resources/mlio_stub/ /miniforge3/lib/python${PYTHON_VERSION}/site-packages/mlio/
RUN cp /miniforge3/lib/python${PYTHON_VERSION}/site-packages/mlio/integ/__init__.py \
  /miniforge3/lib/python${PYTHON_VERSION}/site-packages/mlio/integ/arrow.py \
  && cp /miniforge3/lib/python${PYTHON_VERSION}/site-packages/mlio/integ/__init__.py \
    /miniforge3/lib/python${PYTHON_VERSION}/site-packages/mlio/integ/numpy.py \
  && cp /miniforge3/lib/python${PYTHON_VERSION}/site-packages/mlio/integ/__init__.py \
    /miniforge3/lib/python${PYTHON_VERSION}/site-packages/mlio/integ/scipy.py

# Copy compiled SQLite from builder stage
COPY --from=sqlite-builder /usr/local/bin/sqlite3 /usr/local/bin/sqlite3
COPY --from=sqlite-builder /usr/local/lib/libsqlite3.* /usr/local/lib/
COPY --from=sqlite-builder /usr/local/include/sqlite3*.h /usr/local/include/
RUN ldconfig && echo "/usr/local/lib" >/etc/ld.so.conf.d/sqlite3.conf && ldconfig
ENV PATH="/usr/local/bin:${PATH}"

RUN sqlite3 --version

# Install XGBoost
RUN python3 -m pip install --no-cache-dir -I \
  xgboost==${XGBOOST_VERSION} numpy==2.1.0 pyarrow==22.0.0 pandas==2.2.3

# Starting from XGBoost 2.1.0, the PyPI package won't bundle with nccl
ENV LD_LIBRARY_PATH=/miniforge3/lib/python${PYTHON_VERSION}/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH

# ── Stage: xgboost-sagemaker ───────────────────────────────────────────────
FROM base AS xgboost-sagemaker

ARG SAGEMAKER_XGBOOST_VERSION
ARG PYTHON_VERSION
ARG FRAMEWORK
ARG FRAMEWORK_VERSION
ARG CONTAINER_TYPE

# Install uv for deterministic dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Telemetry
COPY ./scripts/telemetry/deep_learning_container.py /usr/local/bin/deep_learning_container.py
COPY ./scripts/telemetry/bash_telemetry.sh.template /tmp/bash_telemetry.sh.template
COPY ./scripts/common/setup_oss_compliance.sh setup_oss_compliance.sh
RUN chmod +x /usr/local/bin/deep_learning_container.py \
  && sed -e "s/{{FRAMEWORK}}/${FRAMEWORK}/g" \
    -e "s/{{FRAMEWORK_VERSION}}/${FRAMEWORK_VERSION}/g" \
    -e "s/{{CONTAINER_TYPE}}/${CONTAINER_TYPE}/g" \
    /tmp/bash_telemetry.sh.template >/usr/local/bin/bash_telemetry.sh \
  && chmod +x /usr/local/bin/bash_telemetry.sh \
  && rm /tmp/bash_telemetry.sh.template \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/etc/bash.bashrc \
  && bash setup_oss_compliance.sh python3 && rm setup_oss_compliance.sh \
  && rm -rf /root/oss_compliance* \
  && rm -rf /tmp/tmp* \
  && rm -rf /var/lib/apt/lists/*

# Install all SageMaker serving dependencies via uv (lockfile-based, conflict-free)
ENV UV_PROJECT_ENVIRONMENT=/miniforge3
COPY docker/xgboost/pyproject.toml docker/xgboost/uv.lock /tmp/deps/
RUN --mount=type=cache,target=/root/.cache/uv cd /tmp/deps \
  && uv sync --frozen --no-dev --no-install-project --inexact --python /miniforge3/bin/python3 \
  && rm -rf /tmp/deps

# Install smdebug (excluded from uv lock — setup.py requires git-secrets at build time)
RUN python3 -m pip install --no-cache-dir git+https://github.com/awslabs/sagemaker-debugger.git@v1.0.32

# Patch sagemaker-inference decoder.py (CWE-502 fix)
COPY docker/xgboost/resources/patches/decoder.py /miniforge3/lib/python${PYTHON_VERSION}/site-packages/sagemaker_inference/decoder.py

# Patch collections.Mapping deprecation in sagemaker-containers
RUN sed -i 's/collections\.Mapping/collections.abc.Mapping/g' \
  /miniforge3/lib/python${PYTHON_VERSION}/site-packages/sagemaker_containers/_mapping.py

# Install sagemaker_xgboost_container wheel (built in wheel-builder stage, no-deps)
COPY --from=wheel-builder /build/dist/sagemaker_xgboost_container-2.0-py2.py3-none-any.whl /tmp/
RUN python3 -m pip install --no-cache-dir --no-deps /tmp/sagemaker_xgboost_container-2.0-py2.py3-none-any.whl \
  && rm /tmp/sagemaker_xgboost_container-2.0-py2.py3-none-any.whl

# MMS setup
RUN useradd -m model-server \
  && mkdir -p /home/model-server/tmp && chown -R model-server /home/model-server
COPY docker/xgboost/resources/mms/config.properties.tmp /home/model-server
ENV XGBOOST_MMS_CONFIG=/home/model-server/config.properties
RUN mkdir -p /tmp/plugins
COPY docker/xgboost/resources/mms/endpoints-1.0.jar /tmp/plugins
RUN chmod +x /tmp/plugins/endpoints-1.0.jar

# Model directory
RUN mkdir -p /opt/ml/models && chmod +rwx /opt/ml/models

# Dask config
RUN mkdir -p /etc/dask
COPY docker/xgboost/resources/dask_configs.yaml /etc/dask/

# Labels
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

# SageMaker environment
ENV SM_INPUT=/opt/ml/input \
  SM_INPUT_TRAINING_CONFIG_FILE=/opt/ml/input/config/hyperparameters.json \
  SM_INPUT_DATA_CONFIG_FILE=/opt/ml/input/config/inputdataconfig.json \
  SM_CHECKPOINT_CONFIG_FILE=/opt/ml/input/config/checkpointconfig.json \
  NCCL_SOCKET_IFNAME=eth \
  SM_MODEL_DIR=/opt/ml/model \
  SAGEMAKER_TRAINING_MODULE=sagemaker_xgboost_container.training:main \
  SAGEMAKER_SERVING_MODULE=sagemaker_xgboost_container.serving:main \
  TEMP=/home/model-server/tmp

EXPOSE 8080