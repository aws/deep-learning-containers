# HuggingFace PyTorch training DLC for SageMaker, layered on the AWS PyTorch training DLC.
ARG BASE_IMAGE=public.ecr.aws/deep-learning-containers/pytorch:2.13.0-cu133-amzn2023-sagemaker

FROM ${BASE_IMAGE} AS install

ARG FRAMEWORK=huggingface-pytorch-training
ARG FRAMEWORK_VERSION=2.13.0
ARG CONTAINER_TYPE=training
ARG DLC_MAJOR_VERSION=1

LABEL maintainer="Hugging Face"
LABEL dlc_major_version="${DLC_MAJOR_VERSION}"

ARG TRANSFORMERS_VERSION=5.17.0
ARG HUGGINGFACE_HUB_VERSION=1.33.0
ARG HF_XET_VERSION=1.7.0
ARG TRL_VERSION=1.15.0
ARG DIFFUSERS_VERSION=0.41.0
ARG HF_TRANSFER_VERSION=0.1.9
ARG DATASETS_VERSION=5.1.0
ARG ACCELERATE_VERSION=1.15.0
ARG PEFT_VERSION=0.21.2
ARG EVALUATE_VERSION=0.4.6
ARG MATH_VERIFY_VERSION=0.9.0
# vllm pins torch exactly (0.25.0-0.26.0 -> torch 2.11, 0.27.0-0.31.0 -> torch 2.13), so
# VLLM_VERSION must match the base image's torch. trl is installed without its
# [vllm] extra because older trl releases cap vllm below the pinned version (trl
# 1.8 caps it at 0.23); vllm itself provides the extra's remaining deps.
ARG VLLM_VERSION=0.31.0
ARG VENV_PYTHON=/opt/venv/bin/python

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONIOENCODING=UTF-8

# --- HuggingFace training stack ------------------------------------------
# Security floors for ECR findings: soupsieve fixes GHSA-2wc2-fm75-p42x and
# GHSA-836r-79rf-4m37; Pillow 12.3.0 fixes CVE-2026-55379; GitPython 3.1.55
# clears 9 HIGH advisories up to GHSA-94p4-4cq8-9g67; pyasn1 0.6.4 clears
# CVE-2026-59885 and CVE-2026-59886; cryptography 50.0.0 clears CVE-2026-69247
# and CVE-2026-69249; aiohttp 3.14.3 clears CVE-2026-69244.
# uv pip install does not honor the requirements of packages already in the
# base, so restate them: the base's mlflow requires pyarrow<26, and datasets
# would otherwise pull pyarrow 26.
RUN uv pip install --python ${VENV_PYTHON} \
  transformers==${TRANSFORMERS_VERSION} \
  huggingface-hub==${HUGGINGFACE_HUB_VERSION} \
  hf-xet==${HF_XET_VERSION} \
  "hf-transfer==${HF_TRANSFER_VERSION}" \
  "datasets==${DATASETS_VERSION}" \
  "accelerate==${ACCELERATE_VERSION}" \
  "peft==${PEFT_VERSION}" \
  "evaluate==${EVALUATE_VERSION}" \
  "diffusers==${DIFFUSERS_VERSION}" \
  "trl==${TRL_VERSION}" \
  "vllm==${VLLM_VERSION}" \
  "math-verify==${MATH_VERIFY_VERSION}" \
  sentencepiece protobuf einops scipy scikit-learn tensorboard \
  "pillow>=12.3.0" \
  "soupsieve>=2.8.4" \
  "GitPython>=3.1.60" \
  "anyio>=4.14.2" \
  "pyasn1>=0.6.4" \
  "cryptography>=50.0.0" \
  "aiohttp>=3.14.3" \
  "urllib3>=2.8.0" \
  "pyarrow<26" \
  && uv cache clean

# --- Efficiency / quantization kernels -----------------------------------
RUN uv pip install --python ${VENV_PYTHON} \
  deepspeed liger-kernel bitsandbytes \
  && uv cache clean

# git: not in the base, but required for `pip install git+https://...` in user
# requirements.txt (shipped by every legacy HF training DLC).
RUN dnf install -y git && dnf clean all

# Re-template telemetry so it reports this image's framework, not the base's.
# The base's /etc/bashrc hook already sources bash_telemetry.sh — overwriting
# the file is enough.
COPY ./scripts/docker/telemetry/deep_learning_container.py /usr/local/bin/deep_learning_container.py
COPY ./scripts/docker/telemetry/bash_telemetry.sh.template /tmp/bash_telemetry.sh.template
RUN chmod +x /usr/local/bin/deep_learning_container.py \
  && sed -e "s/{{FRAMEWORK}}/${FRAMEWORK}/g" \
    -e "s/{{FRAMEWORK_VERSION}}/${FRAMEWORK_VERSION}/g" \
    -e "s/{{CONTAINER_TYPE}}/${CONTAINER_TYPE}/g" \
    /tmp/bash_telemetry.sh.template >/usr/local/bin/bash_telemetry.sh \
  && chmod +x /usr/local/bin/bash_telemetry.sh \
  && rm /tmp/bash_telemetry.sh.template

ENV DLC_CONTAINER_TYPE=${CONTAINER_TYPE}

# Regenerate OSS compliance docs to cover the HF packages added above.
COPY ./scripts/docker/common/setup_oss_compliance.sh /tmp/setup_oss_compliance.sh
RUN bash /tmp/setup_oss_compliance.sh python3 && rm /tmp/setup_oss_compliance.sh

# OS security refresh on every build (CACHE_REFRESH busts the cache daily —
# same command the base itself runs at its own release time).
ARG CACHE_REFRESH=0
RUN echo "cache refresh: ${CACHE_REFRESH}" \
  && dnf upgrade -y --security --releasever latest \
  && dnf clean all

# HF auto-optimization layer — sourced by the entrypoint to set training perf
# defaults at run time. See scripts/docker/huggingface/pytorch/hf_optimizations.sh.
COPY --chmod=755 ./scripts/docker/huggingface/pytorch/hf_optimizations.sh /usr/local/bin/hf_optimizations.sh

# =====================================================================
# sagemaker — HuggingFace training for SageMaker AI
# =====================================================================
FROM install AS sagemaker

ENV HF_HUB_USER_AGENT_ORIGIN="aws:sagemaker:gpu-cuda:training:hf-pytorch"

COPY --chmod=755 ./scripts/docker/huggingface/pytorch/sagemaker_entrypoint.sh /usr/local/bin/sagemaker_entrypoint.sh
ENTRYPOINT ["/usr/local/bin/sagemaker_entrypoint.sh"]
CMD ["train"]