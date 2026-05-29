ARG FINAL_BASE_IMAGE=763104351884.dkr.ecr.us-west-2.amazonaws.com/sglang:0.5.12-gpu-py312-cu130-ubuntu24.04-sagemaker
FROM ${FINAL_BASE_IMAGE} AS sglang-base

LABEL maintainer="Amazon AI"
LABEL dlc_major_version="1"

ARG HUGGINGFACE_HUB_VERSION=1.15.0
ARG DIFFUSERS_VERSION=0.38.0
ARG HF_XET_VERSION=1.5.0

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends curl unzip \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    huggingface-hub==${HUGGINGFACE_HUB_VERSION} \
    hf-xet==${HF_XET_VERSION} \
    grpcio


FROM sglang-base AS sagemaker
ENV HF_HUB_ENABLE_HF_TRANSFER="1" \
    HF_HUB_USER_AGENT_ORIGIN="aws:sagemaker:gpu-cuda:inference:hf-sglang"

# Copy CUDA compat and entrypoint scripts 
COPY start_cuda_compat.sh /usr/local/bin/start_cuda_compat.sh
COPY sagemaker_entrypoint.sh /usr/local/bin/sagemaker_entrypoint.sh

RUN chmod +x /usr/local/bin/start_cuda_compat.sh \
    && chmod +x /usr/local/bin/sagemaker_entrypoint.sh

RUN HOME_DIR=/root \
    && pip install --upgrade pip requests PTable \
    && curl -o ${HOME_DIR}/oss_compliance.zip https://aws-dlinfra-utilities.s3.amazonaws.com/oss_compliance.zip \
    && unzip ${HOME_DIR}/oss_compliance.zip -d ${HOME_DIR}/ \
    && cp ${HOME_DIR}/oss_compliance/test/testOSSCompliance /usr/local/bin/testOSSCompliance \
    && chmod +x /usr/local/bin/testOSSCompliance \
    && chmod +x ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh \
    && ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh ${HOME_DIR} python3 \
    && rm -rf ${HOME_DIR}/oss_compliance*


######################################################
# Fix ffmpeg vulnerabilities by building from source #
# CVE-2024-35366 / CVE-2024-35367 / CVE-2024-35368   #
######################################################
RUN apt-get purge -y ffmpeg libav* || true && apt-get autoremove -y

ARG FFMPEG_TAG=n8.1

RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        nasm \
        yasm \
        libmp3lame-dev \
        libopus-dev \
        libvorbis-dev \
        libvpx-dev \
        libx264-dev \
        libx265-dev \
    && git clone https://git.ffmpeg.org/ffmpeg.git /tmp/ffmpeg \
    && cd /tmp/ffmpeg \
    && git checkout "${FFMPEG_TAG}" \
    && ./configure \
        --prefix=/usr/local \
        --disable-debug \
        --disable-doc \
        --enable-gpl \
        --enable-libmp3lame \
        --enable-libopus \
        --enable-libvorbis \
        --enable-libvpx \
        --enable-libx264 \
        --enable-libx265 \
    && make -j"$(nproc)" \
    && make install \
    && which ffmpeg \
    && ffmpeg -version \
    && ffprobe -version \
    && rm -rf /tmp/ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN apt-get update \
    && apt-get install -y --only-upgrade libgnutls30t64 \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*


ENTRYPOINT ["/usr/local/bin/sagemaker_entrypoint.sh"]
