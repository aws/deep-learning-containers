# AutoGluon training DLC, layered on the AWS PyTorch training DLC (Amazon Linux 2023).
# CI selects the CPU or CUDA base through BASE_IMAGE; everything below is device-agnostic.
ARG BASE_IMAGE=public.ecr.aws/deep-learning-containers/pytorch:2.13-cu133-amzn2023-sagemaker

FROM ${BASE_IMAGE} AS sagemaker

ARG FRAMEWORK=autogluon
ARG FRAMEWORK_VERSION=1.6.1
ARG CONTAINER_TYPE=training
ARG AUTOGLUON_VERSION=1.6.1
ARG DLC_MAJOR_VERSION=1
ARG DLC_MINOR_VERSION=0
ARG VENV_PYTHON=/opt/venv/bin/python

LABEL maintainer="Amazon AI"
LABEL dlc_major_version="${DLC_MAJOR_VERSION}"
LABEL dlc_minor_version="${DLC_MINOR_VERSION}"
LABEL framework="autogluon"
LABEL framework_version="${AUTOGLUON_VERSION}"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# rsync for Ray file sync, poppler-utils for pdf2image (AutoMM documents).
# AL2023 has no tesseract package, so pytesseract-based AutoMM OCR is unavailable.
RUN dnf install -y rsync poppler-utils && dnf clean all

# autogluon.multimodal caps fsspec at 2025.3; the base's s3fs needs fsspec 2026.x.
# Install AutoGluon, then restore the base's fsspec so s3fs/boto3/SageMaker SDK stay untouched.
RUN BASE_FSSPEC=$(uv pip freeze --python ${VENV_PYTHON} | grep -E '^fsspec==') \
  && uv pip install --python ${VENV_PYTHON} "autogluon==${AUTOGLUON_VERSION}" \
  && uv pip install --python ${VENV_PYTHON} "${BASE_FSSPEC}" \
  && uv cache clean

# Drop Transformer Engine (GPU base only): unused by AutoGluon, and peft imports it at
# load time, which fails without an NVIDIA driver (libcuda.so.1).
RUN TE_PKGS=$(uv pip list --python ${VENV_PYTHON} 2>/dev/null | awk '{print $1}' | grep -i '^transformer-engine' || true) \
  && if [ -n "${TE_PKGS}" ]; then uv pip uninstall --python ${VENV_PYTHON} ${TE_PKGS}; fi

# Fail on unmet requirements, tolerating only autogluon-multimodal's fsspec cap.
# TODO(autogluon 1.6.2): the cap is being lifted upstream; remove the grep -v then.
RUN uv pip check --python ${VENV_PYTHON} >/tmp/uv_pip_check.txt 2>&1 || true \
  && cat /tmp/uv_pip_check.txt \
  && ! grep -F 'requires' /tmp/uv_pip_check.txt | grep -v -F 'autogluon-multimodal` requires `fsspec' \
  && rm /tmp/uv_pip_check.txt

# Telemetry: report this image's framework instead of the base's.
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

# OSS compliance for the added packages, plus AutoGluon's third-party license bundle.
COPY ./scripts/docker/common/setup_oss_compliance.sh /tmp/setup_oss_compliance.sh
RUN bash /tmp/setup_oss_compliance.sh python3 && rm /tmp/setup_oss_compliance.sh \
  && curl -fsSL -o /licenses-autogluon.txt https://autogluon.s3.us-west-2.amazonaws.com/licenses/THIRD-PARTY-LICENSES.txt

# OS security refresh; CACHE_REFRESH busts the cache daily.
ARG CACHE_REFRESH=0
RUN echo "cache refresh: ${CACHE_REFRESH}" \
  && dnf upgrade -y --security --releasever latest \
  && dnf clean all \
  && rm -rf /tmp/tmp* /tmp/pip-* /tmp/uv-*

# ENTRYPOINT / CMD inherited from the base (SageMaker training toolkit launcher).