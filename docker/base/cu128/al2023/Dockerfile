# ============================================================
# Base builder (AL2023 + build tools)
# ============================================================
FROM nvidia/cuda:12.8.1-base-amzn2023 AS base-builder

RUN mv /usr/local/cuda/compat /usr/local \
  && dnf install -y --setopt=install_weak_deps=False \
    automake \
    autoconf \
    ca-certificates \
    cmake \
    curl \
    gcc \
    gcc-c++ \
    git \
    glib2 \
    jq \
    libcurl-devel \
    libffi-devel \
    libtool \
    make \
    numactl \
    numactl-devel \
    openssl \
    openssl-devel \
    pkgconfig \
    tar \
    unzip \
    vim \
    wget \
    xz \
    xz-devel \
    bzip2-devel \
    zlib-devel \
    findutils \
    hwloc-devel \
    libgomp \
    libibverbs-devel \
    mesa-libEGL \
    mesa-libGL \
    libSM \
    libXext \
    libXrender-devel \
    python3-devel \
    check \
    check-devel \
  && dnf clean all \
  && rm -rf /var/cache/dnf

# ============================================================
# Python builder (compile from source)
# ============================================================
FROM base-builder AS python-builder

ARG PYTHON_VERSION="3.12.10"
COPY ./scripts/common/install_python.sh install_python.sh
RUN bash install_python.sh ${PYTHON_VERSION} && rm install_python.sh

COPY ./docker/base/cu128/al2023/pyproject.toml ./docker/base/cu128/al2023/uv.lock /tmp/
RUN cd /tmp && uv sync --frozen --no-dev --system --no-cache && rm pyproject.toml uv.lock

# ============================================================
# CUDA builder (full toolkit + cuDNN + NCCL)
# ============================================================
FROM base-builder AS cuda-builder

ARG CUDA_MAJOR="12"
ARG CUDA_MINOR="8"
COPY ./scripts/common/install_cuda.sh install_cuda.sh
RUN bash install_cuda.sh "${CUDA_MAJOR}.${CUDA_MINOR}" "al2023" && rm install_cuda.sh

# ============================================================
# Final image
# ============================================================
FROM nvidia/cuda:12.8.1-base-amzn2023 AS final

ARG PYTHON="python3"
ARG PYTHON_SHORT_VERSION="3.12"
ARG CUDA_MAJOR="12"
ARG CUDA_MINOR="8"
ARG EFA_VERSION="1.42.0"

LABEL maintainer="Amazon AI"
LABEL dlc_major_version="1"

ENV LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  DLC_CONTAINER_TYPE=base \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONIOENCODING=UTF-8 \
  CUDA_HOME="/usr/local/cuda" \
  PATH="/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:/usr/local/cuda/bin:${PATH}" \
  LD_LIBRARY_PATH="/usr/local/lib:/usr/local/cuda/lib64:/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:${LD_LIBRARY_PATH}"

WORKDIR /

# Copy python from builder — only specific binaries, not uv or unneeded console_scripts
COPY --from=python-builder /usr/local/lib/python${PYTHON_SHORT_VERSION} /usr/local/lib/python${PYTHON_SHORT_VERSION}
COPY --from=python-builder /usr/local/include/python${PYTHON_SHORT_VERSION} /usr/local/include/python${PYTHON_SHORT_VERSION}
COPY --from=python-builder /usr/local/bin/python${PYTHON_SHORT_VERSION} /usr/local/bin/python${PYTHON_SHORT_VERSION}
COPY --from=python-builder /usr/local/bin/pip${PYTHON_SHORT_VERSION} /usr/local/bin/pip${PYTHON_SHORT_VERSION}
COPY --from=python-builder /usr/local/bin/aws /usr/local/bin/aws
RUN ln -sf /usr/local/bin/python${PYTHON_SHORT_VERSION} /usr/local/bin/python3 \
  && ln -sf /usr/local/bin/python${PYTHON_SHORT_VERSION} /usr/local/bin/python \
  && ln -sf /usr/local/bin/pip${PYTHON_SHORT_VERSION} /usr/local/bin/pip3 \
  && ln -sf /usr/local/bin/pip${PYTHON_SHORT_VERSION} /usr/local/bin/pip

# Copy CUDA toolkit, cuDNN, NCCL from builder
COPY --from=cuda-builder /usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR} /usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}

# Copy scripts
COPY ./scripts/common/install_efa_al2023.sh install_efa.sh
COPY ./scripts/telemetry/deep_learning_container.py /usr/local/bin/deep_learning_container.py
COPY ./scripts/telemetry/bash_telemetry.sh.template /tmp/bash_telemetry.sh.template
COPY ./scripts/common/setup_oss_compliance.sh /tmp/setup_oss_compliance.sh
COPY ./scripts/common/start_cuda_compat.sh /usr/local/bin/start_cuda_compat.sh

ARG FRAMEWORK
ARG FRAMEWORK_VERSION
ARG CONTAINER_TYPE

RUN chmod +x /usr/local/bin/deep_learning_container.py \
  && chmod +x /usr/local/bin/start_cuda_compat.sh \
  && sed -e "s/{{FRAMEWORK}}/${FRAMEWORK}/g" \
    -e "s/{{FRAMEWORK_VERSION}}/${FRAMEWORK_VERSION}/g" \
    -e "s/{{CONTAINER_TYPE}}/${CONTAINER_TYPE}/g" \
    /tmp/bash_telemetry.sh.template >/usr/local/bin/bash_telemetry.sh \
  && chmod +x /usr/local/bin/bash_telemetry.sh \
  && rm /tmp/bash_telemetry.sh.template \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/etc/bash.bashrc \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/etc/bashrc \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/root/.bashrc \
  # Install EFA
  && bash install_efa.sh ${EFA_VERSION} \
  && rm install_efa.sh \
  # OSS compliance
  && dnf upgrade -y --security --releasever latest \
  && dnf install -y --setopt=install_weak_deps=False unzip curl \
  && bash /tmp/setup_oss_compliance.sh ${PYTHON} \
  && dnf remove -y unzip curl \
  && dnf clean all && rm -rf /var/cache/dnf \
  && rm /tmp/setup_oss_compliance.sh \
  && rm -rf /root/oss_compliance* \
  && rm -rf /tmp/tmp* \
  && rm -rf /root/.cache || true

CMD ["/bin/bash"]