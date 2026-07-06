# ============================================================================
# Scikit-learn DLC — Ubuntu 24.04 / Python 3.12 (CPU)
# Multi-stage build:
#
#   sqlite-builder ─── (compiles SQLite 3.50.2 from source)
#   wheel-builder ──── (holds pre-built sagemaker_sklearn_container wheel)
#   sklearn-sagemaker  (runtime with SageMaker serving layer)
#
# Based on: github.com/aws/sagemaker-scikit-learn-container/docker/1.4-2-py312
# Wheel is pre-built in CI (scripts/ci/build/sklearn/pre_build.sh) and placed
# at docker/sklearn/1.4-2-py312/prebuilt.whl before docker build.
# ============================================================================

# ── Global ARGs ─────────────────────────────────────────────────────────────
ARG UBUNTU_VERSION=24.04
ARG UBUNTU_IMAGE_DIGEST=b359f1067efa76f37863778f7b6d0e8d911e3ee8efa807ad01fbf5dc1ef9006b
ARG PYTHON_VERSION=3.12
ARG PYARROW_VERSION=17.0.0
ARG NUMPY_VERSION=2.1.0

# ── Stage: sqlite-builder ───────────────────────────────────────────────────
FROM ubuntu:${UBUNTU_VERSION}@sha256:${UBUNTU_IMAGE_DIGEST} AS sqlite-builder
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential wget ca-certificates \
  && cd /tmp \
  && wget https://www.sqlite.org/2025/sqlite-autoconf-3500200.tar.gz \
  && tar xzf sqlite-autoconf-3500200.tar.gz \
  && cd sqlite-autoconf-3500200 \
  && ./configure --prefix=/usr/local && make && make install && ldconfig \
  && cd / && rm -rf /tmp/sqlite-autoconf-* \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Stage: wheel-builder ────────────────────────────────────────────────────
# Wheel is pre-built in CI (scripts/ci/build/sklearn/pre_build.sh) and placed at
# docker/sklearn/1.4-2-py312/prebuilt.whl before docker build starts.
FROM scratch AS wheel-builder
COPY docker/sklearn/1.4-2-py312/prebuilt.whl /build/dist/sagemaker_sklearn_container-2.0-py3-none-any.whl

# ── Stage: sklearn-sagemaker ────────────────────────────────────────────────
FROM ubuntu:${UBUNTU_VERSION}@sha256:${UBUNTU_IMAGE_DIGEST} AS sklearn-sagemaker

ARG PYTHON_VERSION
ARG PYARROW_VERSION
ARG NUMPY_VERSION
ARG FRAMEWORK
ARG FRAMEWORK_VERSION
ARG CONTAINER_TYPE

LABEL maintainer="Amazon AI"
LABEL dlc_major_version="1"

ENV DEBIAN_FRONTEND=noninteractive \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONIOENCODING=UTF-8 \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PIP_ROOT_USER_ACTION=ignore

# System dependencies + Apache Arrow + Python 3.12 (deadsnakes)
RUN apt-get update \
  && apt-get -y upgrade \
  && apt-get -y install --no-install-recommends \
    curl git jq libatlas-base-dev nginx openjdk-8-jdk-headless unzip wget expat tzdata apparmor \
    libgstreamer1.0-0 libxml2 libsqlite3-0 software-properties-common ca-certificates lsb-release \
    build-essential linux-libc-dev \
  && wget https://packages.apache.org/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb \
  && apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb \
  && rm ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb \
  && apt-get update \
  && apt-get install -y -V \
    libarrow-dev=17.0.0-1 libarrow-dataset-dev=17.0.0-1 \
    libparquet-dev=17.0.0-1 libarrow-acero-dev=17.0.0-1 \
  && add-apt-repository ppa:deadsnakes/ppa \
  && apt-get update \
  && apt-get -y install --no-install-recommends \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev \
  && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
  && ln -sf /usr/bin/python3 /usr/bin/python \
  && curl -sS https://bootstrap.pypa.io/get-pip.py | python3 - --break-system-packages \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime \
  && dpkg-reconfigure --frontend noninteractive tzdata

# Install uv for fast Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
  && mv /root/.local/bin/uv /usr/local/bin/uv

ENV PATH=/usr/local/bin:${PATH}

# Copy compiled SQLite from builder stage
COPY --from=sqlite-builder /usr/local/bin/sqlite3 /usr/local/bin/sqlite3
COPY --from=sqlite-builder /usr/local/lib/libsqlite3.* /usr/local/lib/
COPY --from=sqlite-builder /usr/local/include/sqlite3*.h /usr/local/include/
RUN ldconfig \
  && echo "/usr/local/lib" >/etc/ld.so.conf.d/sqlite3.conf \
  && ldconfig \
  && sqlite3 --version

# Install awscli
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
  && unzip awscliv2.zip \
  && ./aws/install \
  && rm -rf aws awscliv2.zip

# Install core scientific packages with exact pinned versions
RUN uv pip install --system --no-cache --break-system-packages \
  numpy==${NUMPY_VERSION} \
  scikit-learn==${FRAMEWORK_VERSION} \
  pyarrow==${PYARROW_VERSION}

# Telemetry
COPY ./scripts/docker/telemetry/deep_learning_container.py /usr/local/bin/deep_learning_container.py
COPY ./scripts/docker/telemetry/bash_telemetry.sh.template /tmp/bash_telemetry.sh.template
COPY ./scripts/docker/common/setup_oss_compliance.sh setup_oss_compliance.sh
RUN chmod +x /usr/local/bin/deep_learning_container.py \
  && sed -e "s/{{FRAMEWORK}}/${FRAMEWORK}/g" \
    -e "s/{{FRAMEWORK_VERSION}}/${FRAMEWORK_VERSION}/g" \
    -e "s/{{CONTAINER_TYPE}}/${CONTAINER_TYPE}/g" \
    /tmp/bash_telemetry.sh.template >/usr/local/bin/bash_telemetry.sh \
  && chmod +x /usr/local/bin/bash_telemetry.sh \
  && rm /tmp/bash_telemetry.sh.template \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/etc/bash.bashrc \
  && bash setup_oss_compliance.sh python${PYTHON_VERSION} && rm setup_oss_compliance.sh \
  && rm -rf /root/oss_compliance* /tmp/tmp*

# Install remaining Python deps from upstream requirements.txt (verbatim)
COPY docker/sklearn/1.4-2-py312/requirements.txt /requirements.txt
RUN uv pip install --system --break-system-packages -r /requirements.txt \
  && rm /requirements.txt

# Patch collections.Mapping deprecation in sagemaker-containers (Python 3.12 compat)
RUN sed -i 's/collections\.Mapping/collections.abc.Mapping/g' \
  /usr/local/lib/python${PYTHON_VERSION}/dist-packages/sagemaker_containers/_mapping.py

# Install sagemaker_sklearn_container wheel (pre-built in CI)
COPY --from=wheel-builder /build/dist/sagemaker_sklearn_container-2.0-py3-none-any.whl /tmp/
RUN uv pip install --system --no-cache --break-system-packages \
  /tmp/sagemaker_sklearn_container-2.0-py3-none-any.whl \
  && rm /tmp/sagemaker_sklearn_container-2.0-py3-none-any.whl

# Force upgrade six to 1.16.0 for six.moves compatibility on py3.12
RUN uv pip install --system --break-system-packages --upgrade --force-reinstall six==1.16.0

# Patch sagemaker-inference decoder.py (CWE-502 fix — disables pickle deserialization)
COPY docker/sklearn/resources/patches/decoder.py /usr/local/lib/python${PYTHON_VERSION}/dist-packages/sagemaker_inference/decoder.py

# Configure pip to always use --break-system-packages for Python 3.12
RUN mkdir -p /root/.config/pip \
  && printf '[global]\nbreak-system-packages = true\n' >/root/.config/pip/pip.conf

# MMS setup
RUN useradd -m model-server \
  && mkdir -p /home/model-server/tmp && chown -R model-server /home/model-server
COPY docker/sklearn/resources/mms/config.properties.tmp /home/model-server
ENV SKLEARN_MMS_CONFIG=/home/model-server/config.properties
RUN mkdir -p /tmp/plugins
COPY docker/sklearn/resources/mms/endpoints-1.0.jar /tmp/plugins
RUN chmod +x /tmp/plugins/endpoints-1.0.jar

# Model directory
RUN mkdir -p /opt/ml/models && chmod +rwx /opt/ml/models

# Drop temp dirs/files left by pip and uv (sanity check requires clean /tmp)
RUN rm -rf /tmp/tmp* /tmp/pip-* /tmp/uv*

# Labels
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

# SageMaker environment
ENV SAGEMAKER_SKLEARN_VERSION=1.4-2-py312 \
  SM_INPUT=/opt/ml/input \
  SM_INPUT_TRAINING_CONFIG_FILE=/opt/ml/input/config/hyperparameters.json \
  SM_INPUT_DATA_CONFIG_FILE=/opt/ml/input/config/inputdataconfig.json \
  SM_CHECKPOINT_CONFIG_FILE=/opt/ml/input/config/checkpointconfig.json \
  SM_MODEL_DIR=/opt/ml/model \
  SAGEMAKER_TRAINING_MODULE=sagemaker_sklearn_container.training:main \
  SAGEMAKER_SERVING_MODULE=sagemaker_sklearn_container.serving:main \
  TEMP=/home/model-server/tmp

EXPOSE 8080

# Security refresh — bust cache to pull latest apt patches on each daily rebuild
ARG CACHE_REFRESH
RUN apt-get update && apt-get upgrade -y \
  && apt-get clean && rm -rf /var/lib/apt/lists/*