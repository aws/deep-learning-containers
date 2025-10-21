FROM docker.io/vllm/vllm-openai:v0.11.0 as base
ARG PYTHON="python3"
LABEL maintainer="Amazon AI"
ARG EFA_VERSION="1.43.3"
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

COPY deep_learning_container.py /usr/local/bin/deep_learning_container.py
COPY bash_telemetry.sh /usr/local/bin/bash_telemetry.sh
COPY install_efa.sh install_efa.sh

RUN chmod +x /usr/local/bin/deep_learning_container.py && \
    chmod +x /usr/local/bin/bash_telemetry.sh && \
    echo 'source /usr/local/bin/bash_telemetry.sh' >> /etc/bash.bashrc  && \
    # OSS compliance - use Python zipfile instead of unzip
    HOME_DIR=/root && \
    curl -o ${HOME_DIR}/oss_compliance.zip https://aws-dlinfra-utilities.s3.amazonaws.com/oss_compliance.zip && \
    python3 -c "import zipfile, os; zipfile.ZipFile('/root/oss_compliance.zip').extractall('/root/'); os.remove('/root/oss_compliance.zip')" && \
    cp ${HOME_DIR}/oss_compliance/test/testOSSCompliance /usr/local/bin/testOSSCompliance && \
    chmod +x /usr/local/bin/testOSSCompliance && \
    chmod +x ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh && \
    ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh ${HOME_DIR} ${PYTHON} && \
    # create symlink for python
    ln -s /usr/bin/python3 /usr/bin/python && \
    # clean up
    rm -rf ${HOME_DIR}/oss_compliance* && \
    rm -rf /tmp/tmp* && \
    rm -rf /tmp/uv* && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache | true


RUN bash install_efa.sh ${EFA_VERSION} && \
    rm install_efa.sh && \
    mkdir -p /tmp/nvjpeg \
    && cd /tmp/nvjpeg \
    && wget https://developer.download.nvidia.com/compute/cuda/redist/libnvjpeg/linux-x86_64/libnvjpeg-linux-x86_64-12.4.0.76-archive.tar.xz \
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

RUN apt-get upgrade && \
    apt-get upgrade -y && \
    apt-get clean

COPY dockerd_entrypoint.sh /usr/local/bin/dockerd_entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd_entrypoint.sh 

ENTRYPOINT ["/usr/local/bin/dockerd_entrypoint.sh"]

# ====================== ray serve =========================================
FROM base AS vllm-rayserve-ec2

RUN uv pip install --system ray[serve]==2.49.0 && \
    uv cache clean \
    apt-get upgrade && \
    apt-get upgrade -y && \
    apt-get clean

COPY dockerd_entrypoint.sh /usr/local/bin/dockerd_entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd_entrypoint.sh 

ENTRYPOINT ["/usr/local/bin/dockerd_entrypoint.sh"]

# ====================== sagemaker =========================================
FROM base AS vllm-sagemaker

RUN apt-get upgrade && \
    apt-get upgrade -y && \
    apt-get clean

COPY sagemaker_entrypoint.sh /usr/local/bin/sagemaker_entrypoint.sh
RUN chmod +x /usr/local/bin/sagemaker_entrypoint.sh

ENTRYPOINT ["/usr/local/bin/sagemaker_entrypoint.sh"]
