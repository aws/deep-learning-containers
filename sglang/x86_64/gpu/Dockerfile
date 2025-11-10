FROM lmsysorg/sglang:v0.5.5-cu129-amd64 AS base

# ====================================================
# ====================== common ======================
# ====================================================

ARG PYTHON="python3"
ARG EFA_VERSION="1.43.3"

LABEL maintainer="Amazon AI"
LABEL dlc_major_version="1"

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DLC_CONTAINER_TYPE=base \
    # Python won’t try to write .pyc or .pyo files on the import of source modules
    # Force stdin, stdout and stderr to be totally unbuffered. Good for logging
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    LD_LIBRARY_PATH="/usr/local/lib:/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:/opt/amazon/openmpi/lib:/opt/amazon/efa/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    PATH="/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:/usr/local/cuda/bin:${PATH}"

WORKDIR /

# Copy artifacts
# ===============
COPY deep_learning_container.py /usr/local/bin/deep_learning_container.py
COPY bash_telemetry.sh /usr/local/bin/bash_telemetry.sh
COPY install_efa.sh install_efa.sh
COPY start_cuda_compat.sh /usr/local/bin/start_cuda_compat.sh

RUN chmod +x /usr/local/bin/deep_learning_container.py \
 && chmod +x /usr/local/bin/bash_telemetry.sh \
 && chmod +x /usr/local/bin/start_cuda_compat.sh

# Install cuda compat
# ====================
# RUN apt-get update \
#  && apt-get -y upgrade --only-upgrade systemd \
#  && apt-get install -y --allow-change-held-packages --no-install-recommends \
#     cuda-compat-12-9 \
#  && rm -rf /var/lib/apt/lists/* \
#  && apt-get clean

# Install EFA and remove vulnerable nvjpeg
# =========================================
RUN bash install_efa.sh ${EFA_VERSION} \
 && rm install_efa.sh \
 && mkdir -p /tmp/nvjpeg \
 && cd /tmp/nvjpeg \
 # latest cu12 libnvjpeg available is cu124
 && wget https://developer.download.nvidia.com/compute/cuda/redist/libnvjpeg/linux-x86_64/libnvjpeg-linux-x86_64-12.4.0.76-archive.tar.xz \
 && tar -xvf libnvjpeg-linux-x86_64-12.4.0.76-archive.tar.xz \
 && rm -rf /usr/local/cuda/targets/x86_64-linux/lib/libnvjpeg* \
 && rm -rf /usr/local/cuda/targets/x86_64-linux/include/nvjpeg.h \
 && cp libnvjpeg-linux-x86_64-12.4.0.76-archive/lib/libnvjpeg* /usr/local/cuda/targets/x86_64-linux/lib/ \
 && cp libnvjpeg-linux-x86_64-12.4.0.76-archive/include/* /usr/local/cuda/targets/x86_64-linux/include/ \
 && rm -rf /tmp/nvjpeg \
 # create symlink for python
 && rm -rf /usr/bin/python \
 && ln -s /usr/bin/python3 /usr/bin/python \
 # remove cuobjdump and nvdisasm
 && rm -rf /usr/local/cuda/bin/cuobjdump* \
 && rm -rf /usr/local/cuda/bin/nvdisasm*

# Run OSS compliance script
# ==========================
RUN echo 'source /usr/local/bin/bash_telemetry.sh' >> /etc/bash.bashrc  \
 # OSS compliance - use Python zipfile instead of unzip
 && HOME_DIR=/root \
 && curl -o ${HOME_DIR}/oss_compliance.zip https://aws-dlinfra-utilities.s3.amazonaws.com/oss_compliance.zip \
 && python3 -c "import zipfile, os; zipfile.ZipFile('/root/oss_compliance.zip').extractall('/root/'); os.remove('/root/oss_compliance.zip')" \
 && cp ${HOME_DIR}/oss_compliance/test/testOSSCompliance /usr/local/bin/testOSSCompliance \
 && chmod +x /usr/local/bin/testOSSCompliance \
 && chmod +x ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh \
 && ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh ${HOME_DIR} ${PYTHON} \
 # clean up
 && rm -rf ${HOME_DIR}/oss_compliance* \
 && rm -rf /tmp/tmp* \
 && rm -rf /tmp/uv* \
 && rm -rf /var/lib/apt/lists/* \
 && rm -rf /root/.cache | true

# =======================================================
# ====================== sagemaker ======================
# =======================================================

FROM base AS vllm-sagemaker

RUN dpkg -l | grep -E "cuda|nvidia|libnv" | awk '{print $2}' | xargs apt-mark hold \
 && apt-get update \
 && apt-get upgrade -y \
 && apt-get clean

COPY sagemaker_entrypoint.sh /usr/local/bin/sagemaker_entrypoint.sh
RUN chmod +x /usr/local/bin/sagemaker_entrypoint.sh

ENTRYPOINT ["/usr/local/bin/sagemaker_entrypoint.sh"]
