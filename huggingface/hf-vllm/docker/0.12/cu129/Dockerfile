ARG FINAL_BASE_IMAGE=763104351884.dkr.ecr.us-east-1.amazonaws.com/vllm:0.12.0-gpu-py312-cu129-ubuntu22.04-sagemaker-v1.0
FROM ${FINAL_BASE_IMAGE} AS vllm-base

LABEL maintainer="Amazon AI"
LABEL dlc_major_version="1"

ARG HUGGINGFACE_HUB_VERSION=0.36.0
ARG HF_XET_VERSION=1.2.0

RUN apt-get update -y \
&& apt-get install -y --no-install-recommends curl unzip \
&& rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip && \
   pip install --no-cache-dir \
     huggingface-hub==${HUGGINGFACE_HUB_VERSION} \
     hf-xet==${HF_XET_VERSION} \
     grpcio


FROM vllm-base AS sagemaker
ENV HF_HUB_ENABLE_HF_TRANSFER="1" \
    HF_HUB_USER_AGENT_ORIGIN="aws:sagemaker:gpu-cuda:inference:hf-vllm"

COPY cuda-compatibility-lib.sh /usr/local/bin/cuda-compatibility-lib.sh
RUN chmod +x /usr/local/bin/cuda-compatibility-lib.sh

RUN set -eux; \
    HOME_DIR=/root; \
    uv pip install --system --upgrade pip requests PTable; \
    curl -o ${HOME_DIR}/oss_compliance.zip https://aws-dlinfra-utilities.s3.amazonaws.com/oss_compliance.zip; \
    unzip ${HOME_DIR}/oss_compliance.zip -d ${HOME_DIR}/; \
    cp ${HOME_DIR}/oss_compliance/test/testOSSCompliance /usr/local/bin/testOSSCompliance; \
    chmod +x /usr/local/bin/testOSSCompliance; \
    chmod +x ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh; \
    ${HOME_DIR}/oss_compliance/generate_oss_compliance.sh ${HOME_DIR} python3; \
    rm -rf ${HOME_DIR}/oss_compliance*


ENTRYPOINT ["/usr/local/bin/sagemaker_entrypoint.sh"]

