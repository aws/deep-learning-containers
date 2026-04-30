# ============================================================================
# XGBoost DLC — Amazon Linux 2023 / Python 3.12 (CUDA 12.9)
# Multi-stage build:
#
#   builder-base ──── (venv + uv deps)
#   wheel-builder ─── (builds sagemaker_xgboost_container wheel)
#   xgboost-sagemaker (runtime with SageMaker serving layer)
#
# ============================================================================

# ── Global ARGs ─────────────────────────────────────────────────────────────
ARG PYTHON_VERSION=3.12
ARG XGBOOST_VERSION=3.2.0
ARG SAGEMAKER_XGBOOST_VERSION=3.0-5

# ── Stage: builder-base ─────────────────────────────────────────────────────
FROM amazonlinux:2023 AS builder-base
ARG PYTHON_VERSION

RUN dnf install -y --allowerasing \
  python${PYTHON_VERSION} python${PYTHON_VERSION}-devel python${PYTHON_VERSION}-pip \
  gcc gcc-c++ make cmake git openssl-devel ninja-build \
  tar xz curl wget \
  && dnf clean all

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT="/opt/venv"
RUN python${PYTHON_VERSION} -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY docker/xgboost/pyproject.toml docker/xgboost/uv.lock /tmp/build/
WORKDIR /tmp/build
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

# ── Stage: wheel-builder ───────────────────────────────────────────────────
FROM amazonlinux:2023 AS wheel-builder
ARG PYTHON_VERSION

RUN dnf install -y --allowerasing \
  python${PYTHON_VERSION} python${PYTHON_VERSION}-pip git \
  && dnf clean all
RUN pip${PYTHON_VERSION} install setuptools wheel
RUN git clone --depth 1 -b upgrade-xgboost-3.2.0-remove-mlio \
  https://github.com/aws/sagemaker-xgboost-container.git /build
WORKDIR /build
RUN python${PYTHON_VERSION} setup.py bdist_wheel --universal

# ── Stage: xgboost-sagemaker ───────────────────────────────────────────────
FROM nvidia/cuda:12.9.1-base-amzn2023 AS xgboost-sagemaker

ARG PYTHON_VERSION
ARG XGBOOST_VERSION
ARG SAGEMAKER_XGBOOST_VERSION
ARG FRAMEWORK
ARG FRAMEWORK_VERSION
ARG CONTAINER_TYPE

LABEL maintainer="Amazon AI"
LABEL dlc_major_version="1"

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONIOENCODING=utf-8 \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8

# System dependencies
RUN dnf update -y && dnf install -y --allowerasing \
  python${PYTHON_VERSION} python${PYTHON_VERSION}-devel \
  gcc gcc-c++ make git curl wget tar gzip unzip jq \
  java-11-amazon-corretto-headless \
  nginx expat libxml2 glib2 libffi zlib zstd \
  openssl-devel libcurl-devel \
  shadow-utils \
  && dnf clean all

# Copy venv from builder
COPY --from=builder-base /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

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
  && bash setup_oss_compliance.sh python${PYTHON_VERSION} && rm setup_oss_compliance.sh \
  && rm -rf /root/oss_compliance* \
  && rm -rf /tmp/tmp*

# Install smdebug
RUN pip install --no-cache-dir git+https://github.com/awslabs/sagemaker-debugger.git@v1.0.32

# Remove build-time packages to reduce CVE surface
RUN dnf remove -y kernel-headers gcc gcc-c++ make cpp || true && dnf clean all

# Patch sagemaker-inference decoder.py (CWE-502 fix)
COPY docker/xgboost/resources/patches/decoder.py /opt/venv/lib64/python${PYTHON_VERSION}/site-packages/sagemaker_inference/decoder.py

# Patch collections.Mapping deprecation in sagemaker-containers
RUN sed -i 's/collections\.Mapping/collections.abc.Mapping/g' \
  /opt/venv/lib64/python${PYTHON_VERSION}/site-packages/sagemaker_containers/_mapping.py

# Install sagemaker_xgboost_container wheel
COPY --from=wheel-builder /build/dist/sagemaker_xgboost_container-2.0-py2.py3-none-any.whl /tmp/
RUN pip install --force-reinstall PyYAML==6.0.1 \
  && pip install --no-cache --no-deps /tmp/sagemaker_xgboost_container-2.0-py2.py3-none-any.whl \
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
  TEMP=/home/model-server/tmp \
  LD_LIBRARY_PATH=/opt/venv/lib64/python${PYTHON_VERSION}/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH}

EXPOSE 8080