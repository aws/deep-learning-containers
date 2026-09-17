# AutoGluon SageMaker DLC, layered on an AWS PyTorch training DLC
# (Amazon Linux 2023). The image supports both SageMaker commands:
# `train` uses the inherited training toolkit and `serve` uses the lightweight
# AutoGluon HTTP adapter below.
#
# BASE_IMAGE is supplied by:
# - .github/config/image/autogluon/1.6-sagemaker-cpu.yml
# - .github/config/image/autogluon/1.6-sagemaker-cuda.yml
ARG BASE_IMAGE

FROM ${BASE_IMAGE} AS sagemaker

ARG REQUIREMENTS_FILE
ARG FRAMEWORK=autogluon
ARG FRAMEWORK_VERSION=1.6.2
ARG CONTAINER_TYPE=general
ARG DLC_MAJOR_VERSION=1
ARG DLC_MINOR_VERSION=0
ARG VENV_PYTHON=/opt/venv/bin/python

LABEL maintainer="Amazon AI"
LABEL dlc_major_version="${DLC_MAJOR_VERSION}"
LABEL dlc_minor_version="${DLC_MINOR_VERSION}"
LABEL framework="autogluon"
LABEL framework_version="${FRAMEWORK_VERSION}"
# Allow SageMaker to select the listening port through SAGEMAKER_BIND_TO_PORT.
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true
# AutoGluon serves one model artifact per container; SageMaker MME is unsupported.
LABEL com.amazonaws.sagemaker.capabilities.multi-models=false

# rsync for Ray file sync, poppler-utils for pdf2image (AutoMM documents).
# AL2023 has no tesseract package, so pytesseract-based AutoMM OCR is unavailable.
RUN dnf install -y rsync poppler-utils && dnf clean all

# REQUIREMENTS_FILE is supplied by the matching CPU/CUDA image config. It
# contains exact pins only for packages added to or changed from that PyTorch
# DLC base.
COPY ${REQUIREMENTS_FILE} /tmp/autogluon-requirements.in
RUN uv pip install --python ${VENV_PYTHON} \
  --no-deps \
  -r /tmp/autogluon-requirements.in \
  && rm /tmp/autogluon-requirements.in \
  && uv cache clean

# Some GPU base images include Transformer Engine. It is unused by AutoGluon,
# and peft imports it at load time, which fails without an NVIDIA driver
# (libcuda.so.1). This is a no-op when the packages are absent, including on CPU.
RUN TE_PKGS=$(uv pip list --python ${VENV_PYTHON} 2>/dev/null | awk '{print $1}' | grep -i '^transformer-engine' || true) \
  && if [ -n "${TE_PKGS}" ]; then uv pip uninstall --python ${VENV_PYTHON} ${TE_PKGS}; fi

RUN uv pip check --python ${VENV_PYTHON}

# A deliberately small SageMaker adapter. User handler code remains in the
# model artifact (/opt/ml/model/code) and keeps the established SageMaker
# inference-function contract used by AutoGluon Cloud.
COPY scripts/docker/autogluon/server.py /opt/autogluon-server/server.py
COPY scripts/docker/autogluon/gunicorn.conf.py /opt/autogluon-server/gunicorn.conf.py
COPY --chmod=755 scripts/docker/autogluon/install_requirements.py /opt/autogluon-server/install_requirements.py
COPY --chmod=755 scripts/docker/autogluon/sagemaker_entrypoint.sh /usr/local/bin/autogluon-sagemaker-entrypoint.sh

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

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/autogluon-sagemaker-entrypoint.sh"]
CMD ["bash"]