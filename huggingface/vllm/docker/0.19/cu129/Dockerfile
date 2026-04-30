ARG FINAL_BASE_IMAGE=763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm:0.19.1-gpu-py312-cu129-ubuntu22.04-sagemaker
FROM ${FINAL_BASE_IMAGE} AS vllm-base

LABEL maintainer="Amazon AI"
LABEL dlc_major_version="1"

ARG TRANSFORMERS_VERSION=5.6.0
ARG HUGGINGFACE_HUB_VERSION=1.12.0
ARG HF_XET_VERSION=1.4.3

RUN apt-get update -y \
&& apt-get install -y --no-install-recommends curl unzip \
&& rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    transformers==${TRANSFORMERS_VERSION} \
    huggingface-hub==${HUGGINGFACE_HUB_VERSION} \
    hf-xet==${HF_XET_VERSION} \
    grpcio>=1.79.3

FROM vllm-base AS sagemaker

ENV HF_HUB_USER_AGENT_ORIGIN="aws:sagemaker:gpu-cuda:inference:hf-vllm"

# Copy CUDA compat and entrypoint scripts 
COPY --chmod=755 start_cuda_compat.sh /usr/local/bin/start_cuda_compat.sh
COPY --chmod=755 sagemaker_entrypoint.sh /usr/local/bin/sagemaker_entrypoint.sh

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

# Fix CVE-2026-33186 by installing mooncake-transfer-engine with a patched gRPC-go version
RUN pip install --no-cache-dir mooncake-transfer-engine==0.3.10.post2

# Fix CVE-2026-35385 / CVE-2026-35414 / CVE-2026-35386
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        openssh-client=1:8.9p1-3ubuntu0.15 \
        openssh-server=1:8.9p1-3ubuntu0.15 \
    && rm -rf /var/lib/apt/lists/*

RUN HOME_DIR=/root \
    && uv pip install --system --upgrade pip requests PTable \
    && curl -o ${HOME_DIR}/oss_compliance.zip https://aws-dlinfra-utilities.s3.amazonaws.com/oss_compliance.zip \
    && unzip ${HOME_DIR}/oss_compliance.zip -d ${HOME_DIR}/ \
    && cp ${HOME_DIR}/oss_compliance/test/testOSSCompliance /usr/local/bin/testOSSCompliance \
    && chmod +x /usr/local/bin/testOSSCompliance \
    && chmod +x ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh \
    && ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh ${HOME_DIR} python3 \
    && rm -rf ${HOME_DIR}/oss_compliance*

ENTRYPOINT ["/usr/local/bin/sagemaker_entrypoint.sh"]
