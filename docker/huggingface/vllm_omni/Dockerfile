# HuggingFace vLLM-Omni inference DLC — layered on the AWS vLLM DLC (AL2023).
#
# Deliberately NOT layered on AWS's vLLM-Omni DLC
# (docker/vllm_omni/Dockerfile.amzn2023): that image compiles vLLM from source
# for AL2023, so its omni version is bound to whatever that build ships
# (0.21.0rc1 on main; PR #6477 raises it to 0.26.0) and layering on it would
# tie our releases to that build's cadence instead of upstream's.
#
# vllm-omni is a pure-Python vLLM plugin: it registers its models through the
# `vllm.general_plugins` entry point and adds `--omni` to vLLM's own argument
# parser (vllm_omni/engine/arg_utils.py::_add_omni_specific_args), so it
# installs onto any vLLM environment of the same minor version. Upstream's own
# container does exactly that — docker/Dockerfile.cuda@v0.26.0 is
# `FROM vllm/vllm-openai:v0.26.0` plus `uv pip install .`, nothing more.
# Layering on the AWS vLLM DLC therefore tracks upstream vllm-omni releases
# without owning a CUDA build, and keeps the AL2023 base AWS security-patches.
#
# Base digest = public.ecr.aws/deep-learning-containers/vllm:server-sagemaker-cuda-v2
# (vllm-server 0.26.0, amzn2023, cu130, py312, dlc 2.2). Pinned by digest, not
# tag: AWS publishes no immutable tag for this release on the v2 SageMaker line
# (v2.2 / v2.2.0 exist only as -soci artifacts, which are not images), and the
# floating v2 tag will advance to the next vLLM minor — which must never
# silently unpair from VLLM_OMNI_VERSION below.
ARG BASE_IMAGE=public.ecr.aws/deep-learning-containers/vllm@sha256:fe9f03d1b5281606d707f666cbdaa8e3afd2725389103599851dabfe2cf56a83

FROM ${BASE_IMAGE} AS base

ARG FRAMEWORK=huggingface-vllm-omni
ARG FRAMEWORK_VERSION=0.26.0
ARG CONTAINER_TYPE=general
ARG DLC_MAJOR_VERSION=1
ARG DLC_MINOR_VERSION=0

LABEL maintainer="Hugging Face"
LABEL dlc_major_version="${DLC_MAJOR_VERSION}"
LABEL dlc_minor_version="${DLC_MINOR_VERSION}"

# Omni-modality system dependencies: espeak-ng for TTS phonemization, ffmpeg for
# audio/video encode and decode. Both live in SPAL (Supplementary Packages for
# Amazon Linux). Same set AWS installs in docker/vllm_omni; libsndfile is
# already present in the vLLM SageMaker base.
RUN dnf upgrade -y --releasever=latest --setopt=install_weak_deps=False system-release \
  && dnf install -y spal-release \
  && dnf install -y --setopt=install_weak_deps=False espeak-ng ffmpeg-free \
  && dnf clean all && rm -rf /var/cache/dnf

ARG VLLM_OMNI_VERSION=0.26.0

# s3tokenizer is a hard runtime import of vllm-omni's CosyVoice3 model code
# (vllm_omni/model_executor/models/cosyvoice3.py) as of 0.26.0 — it was not in
# 0.21.0rc1. Upstream declares it only in the `dev` optional-dependency group,
# so installing vllm-omni does NOT pull it and CosyVoice3 fails at model load
# with `ModuleNotFoundError: No module named 's3tokenizer'`. Installed in the
# same layer as vllm-omni so the torch/vllm assertions below cover it too.
# Same gap AWS hit in docker/vllm_omni (#6477).
ARG S3TOKENIZER_VERSION=0.3.0

# Two build-time assertions, both guarding silent breakage rather than style:
#
# 1. Pairing. vllm-omni's wheel metadata declares no `vllm` requirement at all;
#    it expects a matching vLLM to be installed already, and upstream pairs
#    release X.Y with vLLM X.Y. Nothing enforces that at install time, so a base
#    bump that moves vLLM's minor must fail here instead of producing an image
#    that imports cleanly and then misbehaves at run time.
#
# 2. Immovability of torch/vllm. This install is the one place they could move:
#    vllm-omni pins several deps exactly (accelerate==1.12.0, diffusers==0.38.0)
#    and its requirements/cuda.txt carries a cublas force-reinstall note for the
#    cu12 case. A resolver surprise that swaps the base's source-built
#    vllm (0.26.0+amzn2023) for a PyPI wheel, or moves torch off the image's
#    CUDA runtime, fails the build.
RUN _v() { python3 -c "from importlib.metadata import version; print(version('$1'))"; } \
  && VLLM_BEFORE="$(_v vllm)" \
  && TORCH_BEFORE="$(_v torch)" \
  && BASE_MINOR="$(echo "${VLLM_BEFORE}" | cut -d+ -f1 | cut -d. -f1,2)" \
  && OMNI_MINOR="$(echo "${VLLM_OMNI_VERSION}" | cut -d. -f1,2)" \
  && if [ "${BASE_MINOR}" != "${OMNI_MINOR}" ]; then \
    echo "FATAL: base vLLM ${VLLM_BEFORE} does not pair with vllm-omni ${VLLM_OMNI_VERSION}"; \
    echo "       bump BASE_IMAGE to the vLLM ${OMNI_MINOR} DLC, or VLLM_OMNI_VERSION to ${BASE_MINOR}.x"; \
    exit 1; \
  fi \
  && uv pip install "vllm-omni==${VLLM_OMNI_VERSION}" "s3tokenizer==${S3TOKENIZER_VERSION}" \
  && VLLM_AFTER="$(_v vllm)" \
  && TORCH_AFTER="$(_v torch)" \
  && if [ "${VLLM_BEFORE}" != "${VLLM_AFTER}" ]; then \
    echo "FATAL: vllm moved ${VLLM_BEFORE} -> ${VLLM_AFTER} during vllm-omni install"; exit 1; \
  fi \
  && if [ "${TORCH_BEFORE}" != "${TORCH_AFTER}" ]; then \
    echo "FATAL: torch moved ${TORCH_BEFORE} -> ${TORCH_AFTER} during vllm-omni install"; exit 1; \
  fi \
  && uv cache clean \
  && rm -rf /tmp/uv*

ARG TRANSFORMERS_VERSION=5.10.2
ARG HUGGINGFACE_HUB_VERSION=1.17.0
ARG HF_XET_VERSION=1.5.0

# HF stack installed after vllm-omni so these are the versions that ship.
# vllm-omni 0.26.0 requires `transformers>=5.5.3` with no upper bound: the
# `<5.9.0` cap carried by AWS's 0.21.0rc1 image was dropped upstream once the
# MOSS-TTS-Nano `_keys_to_ignore_on_load_unexpected` incompatibility was handled
# in code (vllm_omni/model_executor/models/utils.py), so 5.10.x is supported.
RUN uv pip install \
  "transformers==${TRANSFORMERS_VERSION}" \
  "huggingface-hub==${HUGGINGFACE_HUB_VERSION}" \
  "hf-xet==${HF_XET_VERSION}" \
  && uv cache clean \
  && rm -rf /tmp/uv*

# Re-template telemetry so it reports this image's framework, not the base's.
# The base already appends the source hook to /etc/bashrc and /root/.bashrc, so
# overwriting bash_telemetry.sh is enough.
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

# HF auto-optimization layer, shared verbatim with the HF vLLM DLC — sourced by
# the entrypoint to set run-time defaults (hf-xet high-performance downloads,
# tokenizers fork guard, allocator, opt-in LMCache/runai-streamer). The omni
# entrypoint changes one default; see the LMCache note there.
COPY --chmod=755 ./scripts/docker/huggingface/vllm/hf_optimizations.sh /usr/local/bin/hf_optimizations.sh

# Regenerate OSS compliance docs so they cover vllm-omni and its dependency tree.
COPY ./scripts/docker/common/setup_oss_compliance.sh /tmp/setup_oss_compliance.sh
RUN bash /tmp/setup_oss_compliance.sh python3 && rm /tmp/setup_oss_compliance.sh

# CUDA forward compatibility for hosts whose NVIDIA driver predates the CUDA 13
# baseline. Read by vLLM itself (vllm/envs.py: VLLM_ENABLE_CUDA_COMPATIBILITY,
# VLLM_CUDA_COMPATIBILITY_PATH); the compat libraries are installed in the base
# via cuda-compat-13-0. Declared here — the vLLM DLC does not set it — so the
# knob is discoverable with `docker inspect`, matching AWS's omni image.
ENV VLLM_ENABLE_CUDA_COMPATIBILITY=0

# =====================================================================
# sagemaker — HuggingFace vLLM-Omni for SageMaker AI
# =====================================================================
FROM base AS huggingface-vllm-omni-sagemaker

ENV HF_HUB_USER_AGENT_ORIGIN="aws:sagemaker:gpu-cuda:inference:hf-vllm-omni"

# OS security refresh on every build (CACHE_REFRESH busts the cache daily).
# cuda-compat is upgraded explicitly: it ships from the NVIDIA repo and is not
# covered by AL2023 security advisories, so --security alone misses it.
ARG CACHE_REFRESH=0
RUN echo "cache refresh: ${CACHE_REFRESH}" \
  && dnf upgrade -y --security --releasever latest --setopt=install_weak_deps=False \
  && dnf upgrade -y --releasever latest cuda-compat-13-0 \
  && dnf clean all && rm -rf /var/cache/dnf /tmp/*

# /invocations router for omni routes (TTS, image, audio, video), reused from the
# AWS omni image rather than reimplemented. PYTHONPATH already contains
# /usr/local/bin (set by the base), which is how --middleware resolves it.
COPY ./scripts/docker/vllm/omni_sagemaker_serve.py /usr/local/bin/omni_sagemaker_serve.py
COPY --chmod=755 ./scripts/docker/huggingface/vllm_omni/sagemaker_entrypoint.sh /usr/local/bin/sagemaker_entrypoint.sh

ENTRYPOINT ["/usr/local/bin/sagemaker_entrypoint.sh"]