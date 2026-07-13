# HuggingFace PyTorch training DLC, layered on the AWS PyTorch training DLC.
# Two targets map onto the base's two flavors: sagemaker | ec2.
ARG BASE_IMAGE=public.ecr.aws/deep-learning-containers/pytorch:2.11.0-cu130-amzn2023-sagemaker

FROM ${BASE_IMAGE} AS install

ARG FRAMEWORK=huggingface-pytorch
ARG FRAMEWORK_VERSION=2.11.0
ARG CONTAINER_TYPE=training
ARG DLC_MAJOR_VERSION=1

LABEL maintainer="Hugging Face"
LABEL dlc_major_version="${DLC_MAJOR_VERSION}"

ARG TRANSFORMERS_VERSION=5.10.2
ARG HUGGINGFACE_HUB_VERSION=1.17.0
ARG HF_XET_VERSION=1.5.0
ARG TRL_VERSION=1.8.0
# vllm >= 0.24 required: earlier versions ship CVE-affected deps (mooncake
# etc.). trl's [vllm] extra caps vllm at 0.23, so trl is installed without it —
# vllm itself provides the extra's remaining deps (fastapi, pydantic, uvicorn).
ARG VLLM_VERSION=0.25.0
ARG VENV_PYTHON=/opt/venv/bin/python

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONIOENCODING=UTF-8

# The ec2 base flavor doesn't ship uv (the sagemaker one does) — bring our own.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# --- HuggingFace training stack ------------------------------------------
# soupsieve floor is a CVE backport (GHSA-2wc2-fm75-p42x, GHSA-836r-79rf-4m37:
# ReDoS in the CSS selector parser), flagged by the ECR scan on this base.
RUN uv pip install --python ${VENV_PYTHON} \
  transformers==${TRANSFORMERS_VERSION} \
  huggingface-hub==${HUGGINGFACE_HUB_VERSION} \
  hf-xet==${HF_XET_VERSION} \
  hf-transfer \
  datasets accelerate peft evaluate \
  "trl==${TRL_VERSION}" \
  "vllm==${VLLM_VERSION}" \
  math-verify \
  sentencepiece protobuf einops scipy scikit-learn tensorboard \
  "soupsieve>=2.8.4" \
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

# =====================================================================
# ec2 — HuggingFace training for EC2 / EKS
# =====================================================================
FROM install AS ec2

ENV HF_HUB_USER_AGENT_ORIGIN="aws:ec2:gpu-cuda:training:hf-pytorch"

COPY --chmod=755 ./scripts/docker/huggingface/pytorch/ec2_entrypoint.sh /usr/local/bin/ec2_entrypoint.sh
ENTRYPOINT ["/usr/local/bin/ec2_entrypoint.sh"]
CMD ["/bin/bash"]