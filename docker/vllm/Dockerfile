# Declare the argument as default to use as input
# base image: https://hub.docker.com/r/vllm/vllm-openai/tags
ARG BASE_IMAGE=vllm/vllm-openai:v0.12.0

# Use input argument as base image
FROM $BASE_IMAGE AS base

# ====================== common =========================================
ARG PYTHON="python3"
LABEL maintainer="Amazon AI"
ARG EFA_VERSION="1.45.1"
LABEL dlc_major_version="1"
ENV DEBIAN_FRONTEND=noninteractive \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  DLC_CONTAINER_TYPE=base \
  # Python won't try to write .pyc or .pyo files on the import of source modules
  # Force stdin, stdout and stderr to be totally unbuffered. Good for logging
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONIOENCODING=UTF-8 \
  LD_LIBRARY_PATH="/usr/local/lib:/opt/amazon/ofi-nccl/lib:/opt/amazon/openmpi/lib:/opt/amazon/efa/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
  PATH="/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:/usr/local/cuda/bin:${PATH}"

WORKDIR /

COPY ./scripts/telemetry/deep_learning_container.py /usr/local/bin/deep_learning_container.py
COPY ./scripts/telemetry/bash_telemetry.sh.template /tmp/bash_telemetry.sh.template
COPY ./scripts/common/setup_oss_compliance.sh setup_oss_compliance.sh

ARG FRAMEWORK
ARG FRAMEWORK_VERSION
ARG CONTAINER_TYPE
RUN chmod +x /usr/local/bin/deep_learning_container.py \
  && sed -e "s/{{FRAMEWORK}}/${FRAMEWORK}/g" \
    -e "s/{{FRAMEWORK_VERSION}}/${FRAMEWORK_VERSION}/g" \
    -e "s/{{CONTAINER_TYPE}}/${CONTAINER_TYPE}/g" \
    /tmp/bash_telemetry.sh.template >/usr/local/bin/bash_telemetry.sh \
  && chmod +x /usr/local/bin/bash_telemetry.sh \
  && rm /tmp/bash_telemetry.sh.template \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/etc/bash.bashrc \
  && bash setup_oss_compliance.sh ${PYTHON} && rm setup_oss_compliance.sh \
  # create symlink for python
  && ln -s /usr/bin/python3 /usr/bin/python \
  # clean up
  && rm -rf ${HOME_DIR}/oss_compliance* \
  && rm -rf /tmp/tmp* \
  && rm -rf /tmp/uv* \
  && rm -rf /var/lib/apt/lists/* \
  && rm -rf /root/.cache | true

COPY ./scripts/common/install_efa.sh install_efa.sh
RUN bash install_efa.sh ${EFA_VERSION} \
  && rm install_efa.sh \
  && mkdir -p /tmp/nvjpeg \
  && cd /tmp/nvjpeg \
  && curl -O https://developer.download.nvidia.com/compute/cuda/redist/libnvjpeg/linux-x86_64/libnvjpeg-linux-x86_64-12.4.0.76-archive.tar.xz \
  && tar -xvf libnvjpeg-linux-x86_64-12.4.0.76-archive.tar.xz \
  && rm -rf /usr/local/cuda/targets/x86_64-linux/lib/libnvjpeg* \
  && rm -rf /usr/local/cuda/targets/x86_64-linux/include/nvjpeg.h \
  && cp libnvjpeg-linux-x86_64-12.4.0.76-archive/lib/libnvjpeg* /usr/local/cuda/targets/x86_64-linux/lib/ \
  && cp libnvjpeg-linux-x86_64-12.4.0.76-archive/include/* /usr/local/cuda/targets/x86_64-linux/include/ \
  && rm -rf /tmp/nvjpeg \
  # remove cuobjdump and nvdisasm
  && rm -rf /usr/local/cuda/bin/cuobjdump* \
  && rm -rf /usr/local/cuda/bin/nvdisasm*

# ====================== ec2 =========================================
FROM base AS vllm-ec2

ARG CACHE_REFRESH=0
RUN dpkg -l | grep -E "cuda|nvidia|libnv" | awk '{print $2}' | xargs apt-mark hold \
  && apt-get update \
  && apt-get upgrade -y \
  && apt-get clean

COPY ./scripts/vllm/dockerd_entrypoint.sh /usr/local/bin/dockerd_entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd_entrypoint.sh

ENTRYPOINT ["/usr/local/bin/dockerd_entrypoint.sh"]

# ====================== ray serve =========================================
FROM base AS vllm-rayserve-ec2

RUN uv pip install --system ray[serve]==2.49.0 \
  && uv cache clean

ARG CACHE_REFRESH=0
RUN dpkg -l | grep -E "cuda|nvidia|libnv" | awk '{print $2}' | xargs apt-mark hold \
  && apt-get update \
  && apt-get upgrade -y \
  && apt-get clean

COPY ./scripts/vllm/dockerd_entrypoint.sh /usr/local/bin/dockerd_entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd_entrypoint.sh

ENTRYPOINT ["/usr/local/bin/dockerd_entrypoint.sh"]

# ====================== sagemaker =========================================
FROM base AS vllm-sagemaker

ARG CACHE_REFRESH=0
RUN dpkg -l | grep -E "cuda|nvidia|libnv" | awk '{print $2}' | xargs apt-mark hold \
  && apt-get update \
  && apt-get upgrade -y \
  && apt-get clean

COPY ./scripts/vllm/sagemaker_entrypoint.sh /usr/local/bin/sagemaker_entrypoint.sh
RUN chmod +x /usr/local/bin/sagemaker_entrypoint.sh

ENTRYPOINT ["/usr/local/bin/sagemaker_entrypoint.sh"]