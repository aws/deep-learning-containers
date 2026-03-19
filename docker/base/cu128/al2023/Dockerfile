# ============================================================
# Python builder (compile from source on cuda-base)
# ============================================================
FROM nvidia/cuda:12.8.1-base-amzn2023 AS python-builder

ARG PYTHON_VERSION="3.12.10"
ARG PYTHON_SHORT_VERSION="3.12"

RUN dnf install -y --setopt=install_weak_deps=False \
  gcc gcc-c++ make tar gzip wget \
  openssl-devel libffi-devel bzip2-devel xz-devel zlib-devel \
  && dnf clean all && rm -rf /var/cache/dnf

COPY ./scripts/common/install_python.sh install_python.sh
RUN --mount=type=cache,target=/root/.cache/pip bash install_python.sh ${PYTHON_VERSION} && rm install_python.sh

# ============================================================
# Runtime: CUDA runtime libs + Python
# ============================================================
FROM nvidia/cuda:12.8.1-runtime-amzn2023 AS runtime

ARG PYTHON_SHORT_VERSION="3.12"

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
RUN ln -sf /usr/local/bin/python${PYTHON_SHORT_VERSION} /usr/local/bin/python3 \
  && ln -sf /usr/local/bin/python${PYTHON_SHORT_VERSION} /usr/local/bin/python \
  && ln -sf /usr/local/bin/pip${PYTHON_SHORT_VERSION} /usr/local/bin/pip3 \
  && ln -sf /usr/local/bin/pip${PYTHON_SHORT_VERSION} /usr/local/bin/pip

COPY ./scripts/common/setup_oss_compliance.sh /tmp/setup_oss_compliance.sh

ARG PYTHON="python3"
RUN dnf upgrade -y --security --releasever latest \
  && dnf install -y --allowerasing --setopt=install_weak_deps=False unzip curl \
  && pip install --no-cache-dir requests \
  && bash /tmp/setup_oss_compliance.sh ${PYTHON} \
  && dnf remove -y unzip \
  && dnf clean all && rm -rf /var/cache/dnf \
  && rm /tmp/setup_oss_compliance.sh \
  && rm -rf /root/oss_compliance* \
  && rm -rf /tmp/tmp* \
  && rm -rf /root/.cache

CMD ["/bin/bash"]

# ============================================================
# Devel: CUDA devel libs + Python + build tools
# ============================================================
FROM nvidia/cuda:12.8.1-devel-amzn2023 AS devel

ARG PYTHON_SHORT_VERSION="3.12"

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
RUN ln -sf /usr/local/bin/python${PYTHON_SHORT_VERSION} /usr/local/bin/python3 \
  && ln -sf /usr/local/bin/python${PYTHON_SHORT_VERSION} /usr/local/bin/python \
  && ln -sf /usr/local/bin/pip${PYTHON_SHORT_VERSION} /usr/local/bin/pip3 \
  && ln -sf /usr/local/bin/pip${PYTHON_SHORT_VERSION} /usr/local/bin/pip

RUN dnf upgrade -y --security --releasever latest \
  && dnf install -y --setopt=install_weak_deps=False \
    automake autoconf cmake gcc gcc-c++ git make tar \
  && dnf clean all && rm -rf /var/cache/dnf

COPY ./scripts/common/setup_oss_compliance.sh /tmp/setup_oss_compliance.sh

ARG PYTHON="python3"
RUN dnf install -y --allowerasing --setopt=install_weak_deps=False unzip curl \
  && pip install --no-cache-dir requests \
  && bash /tmp/setup_oss_compliance.sh ${PYTHON} \
  && dnf remove -y unzip \
  && dnf clean all && rm -rf /var/cache/dnf \
  && rm /tmp/setup_oss_compliance.sh \
  && rm -rf /root/oss_compliance* \
  && rm -rf /tmp/tmp* \
  && rm -rf /root/.cache

CMD ["/bin/bash"]