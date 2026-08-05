ARG CUDA_VERSION=13.2.1
# Short CUDA tag — also the docker/base/<dir> name (cu129, cu130, ...)
ARG CUDA_VERSION_SHORT=cu132
# uv release to copy the binary from (pinned via the config build: block)
ARG UV_VERSION=0.11.33

# ============================================================
# uv binary source (pinned)
# ============================================================
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ============================================================
# Python builder (compile from source on cuda-base)
# ============================================================
FROM nvidia/cuda:${CUDA_VERSION}-base-amzn2023 AS python-builder

ARG PYTHON_VERSION="3.13.12"
ARG PYTHON_SHORT_VERSION="3.13"

RUN dnf install -y --setopt=install_weak_deps=False \
  gcc gcc-c++ make tar gzip wget \
  openssl-devel libffi-devel bzip2-devel xz-devel zlib-devel \
  sqlite-devel readline-devel ncurses-devel \
  && dnf clean all && rm -rf /var/cache/dnf

COPY ./scripts/docker/common/install_python.sh install_python.sh
RUN --mount=type=cache,target=/root/.cache/pip bash install_python.sh ${PYTHON_VERSION} && rm install_python.sh

# ============================================================
# GDRCopy builder (userspace lib only — independent of Python)
# ============================================================
FROM nvidia/cuda:${CUDA_VERSION}-devel-amzn2023 AS gdrcopy-builder

ARG GDRCOPY_VERSION="2.6"

COPY ./scripts/docker/common/install_gdrcopy.sh /tmp/install_gdrcopy.sh
RUN bash /tmp/install_gdrcopy.sh ${GDRCOPY_VERSION} && rm /tmp/install_gdrcopy.sh

# ============================================================
# Runtime: CUDA runtime libs + Python
# ============================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-amzn2023 AS runtime

ARG PYTHON_SHORT_VERSION="3.13"
ARG CUDA_VERSION_SHORT

LABEL maintainer="Amazon AI"
LABEL dlc_major_version="1"
LABEL dlc_minor_version="0"

ENV LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONIOENCODING=UTF-8 \
  CUDA_HOME="/usr/local/cuda" \
  PATH="/usr/local/cuda/bin:${PATH}" \
  LD_LIBRARY_PATH="/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

WORKDIR /

COPY --from=python-builder /usr/local/lib/python${PYTHON_SHORT_VERSION} /usr/local/lib/python${PYTHON_SHORT_VERSION}
COPY --from=python-builder /usr/local/include/python${PYTHON_SHORT_VERSION} /usr/local/include/python${PYTHON_SHORT_VERSION}
COPY --from=python-builder /usr/local/bin/python${PYTHON_SHORT_VERSION} /usr/local/bin/python${PYTHON_SHORT_VERSION}
COPY --from=python-builder /usr/local/bin/pip${PYTHON_SHORT_VERSION} /usr/local/bin/pip${PYTHON_SHORT_VERSION}
COPY --from=python-builder /usr/local/lib/libpython${PYTHON_SHORT_VERSION}* /usr/local/lib/
RUN echo "/usr/local/lib" >/etc/ld.so.conf.d/python.conf && ldconfig
RUN ln -sf /usr/local/bin/python${PYTHON_SHORT_VERSION} /usr/local/bin/python3 \
  && ln -sf /usr/local/bin/python${PYTHON_SHORT_VERSION} /usr/local/bin/python \
  && ln -sf /usr/local/bin/pip${PYTHON_SHORT_VERSION} /usr/local/bin/pip3 \
  && ln -sf /usr/local/bin/pip${PYTHON_SHORT_VERSION} /usr/local/bin/pip

RUN dnf install -y --allowerasing --setopt=install_weak_deps=False curl \
  && dnf upgrade -y --security --releasever latest \
  # Explicitly upgrade cuda-compat-13-2 to pick up fixes for NVIDIA Display Driver CVEs
  && dnf upgrade -y --releasever latest cuda-compat-13-2 \
  && dnf clean all && rm -rf /var/cache/dnf

COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/usr/local
COPY ./docker/base/${CUDA_VERSION_SHORT}/pyproject.toml ./docker/base/${CUDA_VERSION_SHORT}/uv.lock /tmp/deps/
RUN --mount=type=cache,target=/root/.cache/uv cd /tmp/deps \
  && uv sync --frozen --no-dev --no-install-project \
  && rm -rf /tmp/deps /tmp/uv-*.lock

COPY ./scripts/docker/common/setup_oss_compliance.sh /tmp/setup_oss_compliance.sh

ARG PYTHON="python3"
RUN bash /tmp/setup_oss_compliance.sh ${PYTHON} \
  && rm /tmp/setup_oss_compliance.sh \
  && rm -rf /root/oss_compliance* \
  && rm -rf /tmp/tmp* \
  && rm -rf /root/.cache

CMD ["/bin/bash"]

# ============================================================
# Devel: CUDA devel libs + Python + build tools + NCCL/EFA stack
# ============================================================
FROM nvidia/cuda:${CUDA_VERSION}-devel-amzn2023 AS devel

ARG PYTHON_SHORT_VERSION="3.13"
ARG CUDA_VERSION
ARG CUDA_VERSION_SHORT
# NCCL RPM version as published in the NVIDIA cuda-rhel9 repo. The +cudaX.Y
# build suffix is derived from CUDA_VERSION below.
ARG NCCL_VERSION="2.29.7-1"
# EFA installer 1.49.0 vends libfabric 2.4.0amzn5.0 and aws-ofi-nccl 1.20.0.
ARG EFA_VERSION="1.49.0"

LABEL maintainer="Amazon AI"
LABEL dlc_major_version="1"
LABEL dlc_minor_version="0"

ENV LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONIOENCODING=UTF-8 \
  CUDA_HOME="/usr/local/cuda" \
  PATH="/usr/local/cuda/bin:/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:${PATH}" \
  LD_LIBRARY_PATH="/usr/local/lib:/usr/local/cuda/lib64:/opt/amazon/ofi-nccl/lib64:/opt/amazon/openmpi/lib:/opt/amazon/openmpi/lib64:/opt/amazon/efa/lib:/opt/amazon/efa/lib64:${LD_LIBRARY_PATH}"

WORKDIR /

COPY --from=python-builder /usr/local/lib/python${PYTHON_SHORT_VERSION} /usr/local/lib/python${PYTHON_SHORT_VERSION}
COPY --from=python-builder /usr/local/include/python${PYTHON_SHORT_VERSION} /usr/local/include/python${PYTHON_SHORT_VERSION}
COPY --from=python-builder /usr/local/bin/python${PYTHON_SHORT_VERSION} /usr/local/bin/python${PYTHON_SHORT_VERSION}
COPY --from=python-builder /usr/local/bin/pip${PYTHON_SHORT_VERSION} /usr/local/bin/pip${PYTHON_SHORT_VERSION}
COPY --from=python-builder /usr/local/lib/libpython${PYTHON_SHORT_VERSION}* /usr/local/lib/
RUN echo "/usr/local/lib" >/etc/ld.so.conf.d/python.conf && ldconfig
RUN ln -sf /usr/local/bin/python${PYTHON_SHORT_VERSION} /usr/local/bin/python3 \
  && ln -sf /usr/local/bin/python${PYTHON_SHORT_VERSION} /usr/local/bin/python \
  && ln -sf /usr/local/bin/pip${PYTHON_SHORT_VERSION} /usr/local/bin/pip3 \
  && ln -sf /usr/local/bin/pip${PYTHON_SHORT_VERSION} /usr/local/bin/pip

RUN dnf install -y --allowerasing --setopt=install_weak_deps=False \
  curl automake autoconf cmake gcc gcc-c++ git make tar \
  && dnf clean all && rm -rf /var/cache/dnf

# GDRCopy userspace library (kernel module is provided by the host)
COPY --from=gdrcopy-builder /usr/local/lib/libgdrapi.so* /usr/local/lib/
COPY --from=gdrcopy-builder /usr/local/include/gdrapi.h /usr/local/include/gdrconfig.h /usr/local/include/
RUN ldconfig

# NCCL from the NVIDIA cuda-rhel9 repo. The amzn2023 repo shipped in this base
# image does not carry libnccl, so the rhel9 repo is added for the NCCL RPMs.
# Derive the +cudaX.Y build suffix from the image's own CUDA toolkit release.
RUN CUDA_MAJOR_MINOR=$(echo "${CUDA_VERSION}" | cut -d. -f1,2) \
  && echo -e '[cuda-rhel9]\nname=cuda-rhel9\nbaseurl=https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64\nenabled=1\ngpgcheck=1\ngpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA' >/etc/yum.repos.d/cuda-rhel9.repo \
  && dnf install -y --setopt=install_weak_deps=False \
    libnccl-${NCCL_VERSION}+cuda${CUDA_MAJOR_MINOR} \
    libnccl-devel-${NCCL_VERSION}+cuda${CUDA_MAJOR_MINOR} \
  && dnf clean all && rm -rf /var/cache/dnf \
  && ldconfig

# EFA installer — vends libfabric, aws-ofi-nccl (NCCL OFI plugin), OpenMPI, and
# rdma-core, plus SSH config for multi-node jobs.
COPY ./scripts/docker/common/install_efa_amzn2023.sh /tmp/install_efa.sh
RUN bash /tmp/install_efa.sh ${EFA_VERSION} && rm /tmp/install_efa.sh

# Security patch — run after all installers so every OS package is covered
RUN dnf upgrade -y --security --releasever latest \
  # Explicitly upgrade cuda-compat-13-2 to pick up fixes for NVIDIA Display Driver CVEs
  && dnf upgrade -y --releasever latest cuda-compat-13-2 \
  && dnf clean all && rm -rf /var/cache/dnf

COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/usr/local
COPY ./docker/base/${CUDA_VERSION_SHORT}/pyproject.toml ./docker/base/${CUDA_VERSION_SHORT}/uv.lock /tmp/deps/
RUN --mount=type=cache,target=/root/.cache/uv cd /tmp/deps \
  && uv sync --frozen --no-dev --no-install-project \
  && rm -rf /tmp/deps /tmp/uv-*.lock

COPY ./scripts/docker/common/setup_oss_compliance.sh /tmp/setup_oss_compliance.sh

ARG PYTHON="python3"
RUN bash /tmp/setup_oss_compliance.sh ${PYTHON} \
  && rm /tmp/setup_oss_compliance.sh \
  && rm -rf /root/oss_compliance* \
  && rm -rf /tmp/tmp* \
  && rm -rf /root/.cache

EXPOSE 22

CMD ["/bin/bash"]