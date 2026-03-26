# ============================================================================
# XGBoost DLC — Ubuntu 20.04 / Python 3.10 (CUDA 12.6)
# Multi-stage build:
#
#   sqlite-builder ─── (compiles SQLite from source)
#   wheel-builder ──── (builds sagemaker_xgboost_container wheel)
#   base ──────────── (miniforge, MLIO, XGBoost, system deps)
#   xgboost-sagemaker  (SageMaker serving layer on top of base)
#
# Based on: github.com/aws/sagemaker-xgboost-container/docker/3.0-5
# ============================================================================

# ── Global ARGs ─────────────────────────────────────────────────────────────
ARG UBUNTU_VERSION=20.04
ARG CUDA_VERSION=12.6.3
ARG MINIFORGE_VERSION=25.9.1-0
ARG CONDA_CHECKSUM=07f64c1d908ae036e9f6a81f97704899311c0ae677d83980d664b9781d4cc5fc
ARG PYTHON_VERSION=3.10
ARG PYARROW_VERSION=22.0.0
ARG MLIO_VERSION=0.9.0
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
ARG CONDA_CHECKSUM
ARG PYTHON_VERSION
ARG PYARROW_VERSION
ARG MLIO_VERSION
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
  && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \
  && dpkg -i cuda-keyring_1.0-1_all.deb \
  && apt-get update && apt-get -y upgrade \
  && apt-get -y install --no-install-recommends \
    build-essential curl git jq libatlas-base-dev expat nginx \
    openjdk-8-jdk-headless unzip wget apparmor linux-libc-dev \
    libxml2 libgstreamer1.0-0 libglib2.0-0 \
  && wget http://es.archive.ubuntu.com/ubuntu/pool/main/libf/libffi/libffi7_3.3-4_amd64.deb \
  && dpkg -i libffi7_3.3-4_amd64.deb \
  && apt-get -y install --no-install-recommends \
    apt-transport-https ca-certificates gnupg software-properties-common \
  && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
  | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
  && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ bionic main' \
  | tee /etc/apt/sources.list.d/kitware.list >/dev/null \
  && apt-get update && rm /usr/share/keyrings/kitware-archive-keyring.gpg \
  && apt-get install -y --no-install-recommends \
    autoconf automake cmake cmake-data doxygen kitware-archive-keyring \
    libcurl4-openssl-dev libssl-dev libtool ninja-build \
    python3-dev python3-distutils python3-pip zlib1g-dev libxml2 zstd libsqlite3-0 \
  && python3 -m pip install --upgrade "pip<=25.2" \
  && python3 -m pip install --upgrade certifi \
  && mkdir -p /etc/apt/keyrings \
  && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
  | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
  && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" \
  | tee /etc/apt/sources.list.d/nodesource.list \
  && apt-get update && apt-get install -y nodejs && npm install -g npm@latest \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Miniforge (conda-forge by default, no Anaconda license required)
RUN curl -L --output /tmp/Miniforge3.sh \
  https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh \
  && echo "${CONDA_CHECKSUM} /tmp/Miniforge3.sh" | sha256sum -c - \
  && bash /tmp/Miniforge3.sh -bfp /miniconda3 \
  && rm /tmp/Miniforge3.sh

ENV PATH=/miniconda3/bin:${PATH}

# Conda packages
RUN conda config --system --set auto_update_conda false \
  && conda install -y --solver=libmamba \
    python=${PYTHON_VERSION} \
    requests=2.32.3 \
    openssl=3.6.1 \
    pyarrow=${PYARROW_VERSION} \
  && conda clean -afy

# Build MLIO from source with Arrow integration
RUN git clone -b v1.65.4 https://github.com/grpc/grpc.git \
  && LIBGRPC_DIR=$(find /miniconda3/pkgs -name "libgrpc-*" -type d | head -n 1) \
  && mkdir -p ${LIBGRPC_DIR}/info/test/examples \
  && cp -r grpc/examples/* ${LIBGRPC_DIR}/info/test/examples/ \
  && cd ${LIBGRPC_DIR}/info/test/examples/node \
  && npm cache clean --force \
  && npm install minimist@latest protobufjs@latest \
  && apt-get purge -y nodejs npm && apt-get autoremove -y \
  && rm -rf /etc/apt/sources.list.d/nodesource.list \
    /etc/apt/keyrings/nodesource.gpg \
    /etc/apt/sources.list.d/kitware.list \
  && apt-get clean && rm -rf /var/lib/apt/lists/* \
  && cd /tmp \
  && git clone --branch v${MLIO_VERSION} https://github.com/awslabs/ml-io.git mlio \
  && cd mlio \
  && sed -i 's/find_package(Arrow 14.0.1 REQUIRED/find_package(Arrow 22.0.0 REQUIRED/g' CMakeLists.txt \
  && sed -i 's/pyarrow==14.0.1/pyarrow==22.0.0/g' src/mlio-py/setup.py \
  && build-tools/build-dependency build/third-party all \
  && mkdir -p build/release && cd build/release \
  && cmake -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_PREFIX_PATH="$(pwd)/../third-party" ../.. \
  && cmake --build . && cmake --build . --target install \
  && cmake -DMLIO_INCLUDE_PYTHON_EXTENSION=ON -DPYTHON_EXECUTABLE="/miniconda3/bin/python3" \
    -DMLIO_INCLUDE_ARROW_INTEGRATION=ON ../.. \
  && cmake --build . --target mlio-py && cmake --build . --target mlio-arrow \
  && cd ../../src/mlio-py \
  && python3 setup.py bdist_wheel \
  && python3 -m pip install typing \
  && python3 -m pip install --upgrade pip==25.2 \
  && python3 -m pip install dist/*.whl \
  && cp -r /tmp/mlio/build/third-party/lib/libtbb* /usr/local/lib/ \
  && ldconfig && rm -rf /tmp/mlio

# Copy compiled SQLite from builder stage
COPY --from=sqlite-builder /usr/local/bin/sqlite3 /usr/local/bin/sqlite3
COPY --from=sqlite-builder /usr/local/lib/libsqlite3.* /usr/local/lib/
COPY --from=sqlite-builder /usr/local/include/sqlite3*.h /usr/local/include/
RUN ldconfig && echo "/usr/local/lib" >/etc/ld.so.conf.d/sqlite3.conf && ldconfig
ENV PATH="/usr/local/bin:${PATH}"

RUN sqlite3 --version

# Install XGBoost
RUN python3 -m pip install --no-cache -I \
  xgboost==${XGBOOST_VERSION} numpy==2.1.0 pyarrow==22.0.0 pandas==2.2.3

ENV LD_LIBRARY_PATH=/miniconda3/lib/python${PYTHON_VERSION}/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH

# ── Stage: xgboost-sagemaker ───────────────────────────────────────────────
FROM base AS xgboost-sagemaker

ARG SAGEMAKER_XGBOOST_VERSION
ARG PYTHON_VERSION
ARG FRAMEWORK
ARG FRAMEWORK_VERSION
ARG CONTAINER_TYPE

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

# Install smdebug
RUN python3 -m pip install --no-cache-dir git+https://github.com/awslabs/sagemaker-debugger.git@v1.0.32

# Install serving dependencies from requirements.txt
COPY docker/xgboost/requirements.txt /requirements.txt
RUN python3 -m pip install --no-cache-dir -r /requirements.txt && rm /requirements.txt

# Patch sagemaker-inference decoder.py (CWE-502 fix)
COPY docker/xgboost/resources/patches/decoder.py /miniconda3/lib/python${PYTHON_VERSION}/site-packages/sagemaker_inference/decoder.py

# Security patches
RUN python3 -m pip install --no-cache-dir "protobuf>=3.20.0,<=3.20.3" "fonttools>=4.60.2"

# Patch collections.Mapping deprecation in sagemaker-containers
RUN sed -i 's/collections\.Mapping/collections.abc.Mapping/g' \
  /miniconda3/lib/python${PYTHON_VERSION}/site-packages/sagemaker_containers/_mapping.py

# Install sagemaker_xgboost_container wheel
COPY --from=wheel-builder /build/dist/sagemaker_xgboost_container-2.0-py2.py3-none-any.whl /tmp/
RUN python3 -m pip install --force-reinstall PyYAML==6.0.1 \
  && python3 -m pip install --no-cache --no-deps /tmp/sagemaker_xgboost_container-2.0-py2.py3-none-any.whl \
  && python3 -m pip uninstall -y typing \
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