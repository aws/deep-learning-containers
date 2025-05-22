ARG PYTHON="python3"
ARG PYTHON_VERSION="3.12.10"
ARG PYTHON_SHORT_VERSION="3.12"
ARG CUDA_MAJOR="12"
ARG CUDA_MINOR="8"
ARG EFA_VERSION="1.40.0"
FROM nvidia/cuda:12.8.1-base-ubuntu24.04 AS base-builder

RUN apt-get update \
    && apt-get -y upgrade --only-upgrade systemd \
    && apt-get install -y --allow-change-held-packages --no-install-recommends \
        automake \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        emacs \
        git \
        jq \
        libcurl4-openssl-dev \
        libglib2.0-0 \
        libegl1 \
        libgl1 \
        libsm6 \
        libssl-dev \
        libxext6 \
        libxrender-dev \
        zlib1g-dev \
        unzip \
        vim \
        wget \
        libhwloc-dev \
        libgomp1 \
        libibverbs-dev \
        libnuma1 \
        libnuma-dev \
        libtool \
        openssl \
        python3-dev \
        autoconf \
        pkg-config \
        check \
        libsubunit0 \
        libsubunit-dev \
        libffi-dev \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

##############################################################################
FROM base-builder AS python-builder
ARG PYTHON_VERSION
COPY install_python.sh install_python.sh
RUN bash install_python.sh ${PYTHON_VERSION} && rm install_python.sh

##############################################################################
FROM base-builder AS cuda-builder
ARG CUDA_MAJOR
ARG CUDA_MINOR
COPY install_cuda.sh install_cuda.sh
RUN bash install_cuda.sh "${CUDA_MAJOR}.${CUDA_MINOR}" && rm install_cuda.sh

##############################################################################
FROM nvidia/cuda:12.8.1-base-ubuntu24.04 AS final 
ARG PYTHON
ARG PYTHON_SHORT_VERSION
ARG CUDA_MAJOR
ARG CUDA_MINOR
ARG EFA_VERSION
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
    CUDA_HOME="/usr/local/cuda" \
    PATH="/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/lib:/usr/local/cuda/lib64:/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:${LD_LIBRARY_PATH}"

WORKDIR /

# + python and pip packages (awscli, boto3, requests)
COPY --from=python-builder /usr/local/lib/python${PYTHON_SHORT_VERSION} /usr/local/lib/python${PYTHON_SHORT_VERSION}
COPY --from=python-builder /usr/local/include/python${PYTHON_SHORT_VERSION} /usr/local/include/python${PYTHON_SHORT_VERSION}
COPY --from=python-builder /usr/local/bin /usr/local/bin
# + cuda-toolkit, cudnn, nccl
COPY --from=cuda-builder /usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR} /usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}
COPY install_efa.sh install_efa.sh 
COPY deep_learning_container.py /usr/local/bin/deep_learning_container.py
RUN chmod +x /usr/local/bin/deep_learning_container.py && \
    # Install EFA
    bash install_efa.sh ${EFA_VERSION} && \
    rm install_efa.sh && \
    # OSS compliance
    apt-get update && \
    apt-get install -y --allow-change-held-packages --no-install-recommends unzip && \
    HOME_DIR=/root && \
    curl -o ${HOME_DIR}/oss_compliance.zip https://aws-dlinfra-utilities.s3.amazonaws.com/oss_compliance.zip && \
    unzip ${HOME_DIR}/oss_compliance.zip -d ${HOME_DIR}/ && \
    cp ${HOME_DIR}/oss_compliance/test/testOSSCompliance /usr/local/bin/testOSSCompliance && \
    chmod +x /usr/local/bin/testOSSCompliance && \
    chmod +x ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh && \
    ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh ${HOME_DIR} ${PYTHON} && \
    rm -rf ${HOME_DIR}/oss_compliance* && \
    rm -rf /tmp/tmp* && \
    rm -rf /root/.cache | true
    
CMD ["/bin/bash"]