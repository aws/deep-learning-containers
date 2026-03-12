# ============================================================
# Lambda Python sources
# We take only the Python binary, stdlib, and lib-dynload.
# Site-packages are intentionally excluded — all dependencies
# (including awslambdaric) are installed via uv so we own the
# full dependency tree with no Lambda-bundled extras.
# ============================================================
FROM public.ecr.aws/lambda/python:3.13 as lambda-python

# ============================================================
# Lambda RIE (downloaded once, copied into runtime stages)
# ============================================================
FROM public.ecr.aws/lambda/python:3.12 as rie-downloader
RUN curl -Lo /usr/local/bin/aws-lambda-rie \
  https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie \
  && chmod +x /usr/local/bin/aws-lambda-rie

# ============================================================
# Shared builder base (CUDA runtime + uv)
# ============================================================
FROM nvidia/cuda:12.8.1-runtime-amzn2023 as builder-base
RUN dnf install -y --setopt=install_weak_deps=False tar gzip findutils \
  && dnf clean all && rm -rf /var/cache/dnf \
  && curl -LsSf https://astral.sh/uv/install.sh | sh \
  && mv /root/.local/bin/uv /usr/local/bin/uv

# ============================================================
# Shared builder base — devel (CUDA devel + uv + git)
# ============================================================
FROM nvidia/cuda:12.8.1-devel-amzn2023 as builder-base-devel
RUN dnf install -y --setopt=install_weak_deps=False tar gzip git findutils \
  && dnf clean all && rm -rf /var/cache/dnf \
  && curl -LsSf https://astral.sh/uv/install.sh | sh \
  && mv /root/.local/bin/uv /usr/local/bin/uv

# ============================================================
# FFmpeg builder (compiled from source with shared libs)
# ============================================================
FROM nvidia/cuda:12.8.1-devel-amzn2023 as ffmpeg-builder
RUN dnf install -y --setopt=install_weak_deps=False \
  tar xz gcc gcc-c++ make cmake pkgconfig nasm yasm \
  && dnf clean all && rm -rf /var/cache/dnf \
  && cd /tmp \
  && curl -L https://ffmpeg.org/releases/ffmpeg-6.1.tar.xz | tar xJ \
  && cd ffmpeg-6.1 \
  && ./configure \
    --prefix=/usr/local \
    --disable-static \
    --enable-shared \
    --disable-doc \
    --disable-htmlpages \
    --disable-manpages \
    --disable-podpages \
    --disable-txtpages \
    --disable-debug \
  && make -j$(nproc) \
  && make install \
  && strip /usr/local/bin/ffmpeg /usr/local/bin/ffprobe \
  && cd /tmp && rm -rf ffmpeg-6.1

# ============================================================
# Base builder (Python 3.13 — awslambdaric, requests, pip-licenses)
# ============================================================
FROM builder-base as builder-base-py3
COPY --from=lambda-python /var/lang/bin /var/lang/bin
COPY --from=lambda-python /var/lang/include /var/lang/include
COPY --from=lambda-python /var/lang/lib /var/lang/lib
RUN rm -rf /var/lang/lib/python3.13/site-packages
ENV PATH="/var/lang/bin:$PATH"
COPY ./docker/lambda/requirements-base.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --python /var/lang/bin/python3.13 -r /tmp/requirements.txt

# ============================================================
# CuPy builder (Python 3.13 — awslambdaric, CuPy, NumPy, SciPy, etc.)
# ============================================================
FROM builder-base as builder-cupy-py3
COPY --from=lambda-python /var/lang/bin /var/lang/bin
COPY --from=lambda-python /var/lang/include /var/lang/include
COPY --from=lambda-python /var/lang/lib /var/lang/lib
RUN rm -rf /var/lang/lib/python3.13/site-packages
ENV PATH="/var/lang/bin:$PATH"
COPY ./docker/lambda/requirements-cupy.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
  uv pip install --python /var/lang/bin/python3.13 -r /tmp/requirements.txt \
  && uv pip check --python /var/lang/bin/python3.13 \
  && (find /var/lang -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true) \
  && find /var/lang -type f -name "*.pyc" -delete \
  && find /var/lang -type f -name "*.pyo" -delete

# ============================================================
# PyTorch builder (Python 3.13 — awslambdaric, PyTorch, SAM2, transformers)
# ============================================================
FROM builder-base-devel as builder-pytorch-py3
COPY --from=lambda-python /var/lang/bin /var/lang/bin
COPY --from=lambda-python /var/lang/include /var/lang/include
COPY --from=lambda-python /var/lang/lib /var/lang/lib
RUN rm -rf /var/lang/lib/python3.13/site-packages
ENV PATH="/var/lang/bin:$PATH"
COPY ./docker/lambda/requirements-pytorch.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
  uv pip install --index-strategy unsafe-best-match --python /var/lang/bin/python3.13 -r /tmp/requirements.txt \
  && uv pip check --python /var/lang/bin/python3.13 \
  && (find /var/lang -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true) \
  && (find /var/lang -type d -name tests -exec rm -rf {} + 2>/dev/null; true) \
  && find /var/lang -type f -name "*.pyc" -delete \
  && find /var/lang -type f -name "*.pyo" -delete

# ============================================================
# Runtime: Base Python 3.13 (CUDA + Python + RIC + RIE)
# ============================================================
FROM nvidia/cuda:12.8.1-runtime-amzn2023 as base-py3
LABEL maintainer="Amazon AI"
LABEL dlc_major_version="1"
RUN dnf upgrade -y --security --releasever latest && dnf clean all && rm -rf /var/cache/dnf
COPY --from=builder-base-py3 /var/lang /var/lang
COPY --from=lambda-python /var/runtime /var/runtime
COPY --from=rie-downloader /usr/local/bin/aws-lambda-rie /usr/local/bin/aws-lambda-rie
COPY ./scripts/lambda/lambda_entrypoint.sh /lambda_entrypoint.sh
RUN ln -sf /var/lang/bin/python3.13 /var/lang/bin/python3 \
  && ln -sf /var/lang/bin/python3 /var/lang/bin/python
ENV PATH="/var/lang/bin:$PATH" \
  LD_LIBRARY_PATH="/var/lang/lib:/lib64:/usr/lib64:/var/runtime:/var/runtime/lib:/var/task:/var/task/lib:/opt/lib:/usr/local/cuda/lib64:/x86_64-bottlerocket-linux-gnu/sys-root/usr/lib/nvidia" \
  PYTHONPATH="/var/runtime" \
  PYTHONDONTWRITEBYTECODE=1 \
  DLC_CONTAINER_TYPE=general \
  LAMBDA_TASK_ROOT=/var/task \
  LAMBDA_RUNTIME_DIR=/var/runtime \
  LANG=en_US.UTF-8 \
  TZ=:/etc/localtime
COPY ./scripts/telemetry/deep_learning_container.py /usr/local/bin/deep_learning_container.py
COPY ./scripts/telemetry/bash_telemetry.sh.template /tmp/bash_telemetry.sh.template
COPY ./scripts/common/setup_oss_compliance.sh /tmp/setup_oss_compliance.sh
ARG FRAMEWORK="lambda"
ARG CONTAINER_TYPE="general"
RUN chmod +x /usr/local/bin/deep_learning_container.py \
  && sed -e "s/{{FRAMEWORK}}/${FRAMEWORK}/g" \
    -e "/{{FRAMEWORK_VERSION}}/d" \
    -e "s/{{CONTAINER_TYPE}}/${CONTAINER_TYPE}/g" \
    /tmp/bash_telemetry.sh.template >/usr/local/bin/bash_telemetry.sh \
  && chmod +x /usr/local/bin/bash_telemetry.sh \
  && rm /tmp/bash_telemetry.sh.template \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/etc/bash.bashrc \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/etc/bashrc \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/root/.bashrc \
  && bash /tmp/setup_oss_compliance.sh python3 \
  && rm /tmp/setup_oss_compliance.sh \
  && rm -rf /var/lang/lib/python3.13/site-packages/pip \
    /var/lang/lib/python3.13/site-packages/pip-25.3.dist-info \
    /var/lang/lib/python3.13/ensurepip/_bundled
WORKDIR /var/task
ENTRYPOINT ["/lambda_entrypoint.sh", "python", "-m", "awslambdaric"]
CMD ["handler.handler"]

# ============================================================
# Runtime: CuPy Python 3.13 (base + CuPy, NumPy, SciPy)
# ============================================================
FROM nvidia/cuda:12.8.1-runtime-amzn2023 as cupy-py3
LABEL maintainer="Amazon AI"
LABEL dlc_major_version="1"
RUN dnf upgrade -y --security --releasever latest && dnf clean all && rm -rf /var/cache/dnf
COPY --from=builder-cupy-py3 /var/lang /var/lang
COPY --from=lambda-python /var/runtime /var/runtime
COPY --from=rie-downloader /usr/local/bin/aws-lambda-rie /usr/local/bin/aws-lambda-rie
COPY ./scripts/lambda/lambda_entrypoint.sh /lambda_entrypoint.sh
RUN ln -sf /var/lang/bin/python3.13 /var/lang/bin/python3 \
  && ln -sf /var/lang/bin/python3 /var/lang/bin/python
ENV PATH="/var/lang/bin:$PATH" \
  LD_LIBRARY_PATH="/var/lang/lib:/lib64:/usr/lib64:/var/runtime:/var/runtime/lib:/var/task:/var/task/lib:/opt/lib:/usr/local/cuda/lib64:/x86_64-bottlerocket-linux-gnu/sys-root/usr/lib/nvidia" \
  PYTHONPATH="/var/runtime" \
  PYTHONDONTWRITEBYTECODE=1 \
  DLC_CONTAINER_TYPE=general \
  LAMBDA_TASK_ROOT=/var/task \
  LAMBDA_RUNTIME_DIR=/var/runtime \
  LANG=en_US.UTF-8 \
  TZ=:/etc/localtime
COPY ./scripts/telemetry/deep_learning_container.py /usr/local/bin/deep_learning_container.py
COPY ./scripts/telemetry/bash_telemetry.sh.template /tmp/bash_telemetry.sh.template
COPY ./scripts/common/setup_oss_compliance.sh /tmp/setup_oss_compliance.sh
ARG FRAMEWORK="lambda"
ARG CONTAINER_TYPE="general"
RUN chmod +x /usr/local/bin/deep_learning_container.py \
  && sed -e "s/{{FRAMEWORK}}/${FRAMEWORK}/g" \
    -e "/{{FRAMEWORK_VERSION}}/d" \
    -e "s/{{CONTAINER_TYPE}}/${CONTAINER_TYPE}/g" \
    /tmp/bash_telemetry.sh.template >/usr/local/bin/bash_telemetry.sh \
  && chmod +x /usr/local/bin/bash_telemetry.sh \
  && rm /tmp/bash_telemetry.sh.template \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/etc/bash.bashrc \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/etc/bashrc \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/root/.bashrc \
  && bash /tmp/setup_oss_compliance.sh python3 \
  && rm /tmp/setup_oss_compliance.sh \
  && rm -rf /var/lang/lib/python3.13/site-packages/pip \
    /var/lang/lib/python3.13/site-packages/pip-25.3.dist-info \
    /var/lang/lib/python3.13/ensurepip/_bundled
WORKDIR /var/task
ENTRYPOINT ["/lambda_entrypoint.sh", "python", "-m", "awslambdaric"]
CMD ["handler.handler"]

# ============================================================
# Runtime: PyTorch Python 3.13 (base + PyTorch, FFmpeg, SAM2)
# ============================================================
FROM nvidia/cuda:12.8.1-runtime-amzn2023 as pytorch-py3
LABEL maintainer="Amazon AI"
LABEL dlc_major_version="1"
RUN dnf upgrade -y --security --releasever latest \
  && dnf install -y --setopt=install_weak_deps=False \
    libxcb libX11 libXext libXfixes alsa-lib \
  && dnf clean all && rm -rf /var/cache/dnf
COPY --from=builder-pytorch-py3 /var/lang /var/lang
COPY --from=lambda-python /var/runtime /var/runtime
COPY --from=ffmpeg-builder /usr/local/bin/ffmpeg /usr/local/bin/ffprobe /usr/local/bin/
COPY --from=ffmpeg-builder /usr/local/lib/libav*.so* /usr/local/lib/libsw*.so* /usr/local/lib/libpostproc.so* /usr/local/lib/
COPY --from=rie-downloader /usr/local/bin/aws-lambda-rie /usr/local/bin/aws-lambda-rie
COPY ./scripts/lambda/lambda_entrypoint.sh /lambda_entrypoint.sh
RUN ln -sf /var/lang/bin/python3.13 /var/lang/bin/python3 \
  && ln -sf /var/lang/bin/python3 /var/lang/bin/python
ENV PATH="/var/lang/bin:/usr/local/bin:$PATH" \
  LD_LIBRARY_PATH="/var/lang/lib:/lib64:/usr/lib64:/var/runtime:/var/runtime/lib:/var/task:/var/task/lib:/opt/lib:/usr/local/cuda/lib64:/usr/local/lib:/x86_64-bottlerocket-linux-gnu/sys-root/usr/lib/nvidia" \
  PYTHONPATH="/var/runtime" \
  PYTHONDONTWRITEBYTECODE=1 \
  DLC_CONTAINER_TYPE=general \
  LAMBDA_TASK_ROOT=/var/task \
  LAMBDA_RUNTIME_DIR=/var/runtime \
  LANG=en_US.UTF-8 \
  TZ=:/etc/localtime
COPY ./scripts/telemetry/deep_learning_container.py /usr/local/bin/deep_learning_container.py
COPY ./scripts/telemetry/bash_telemetry.sh.template /tmp/bash_telemetry.sh.template
COPY ./scripts/common/setup_oss_compliance.sh /tmp/setup_oss_compliance.sh
ARG FRAMEWORK="lambda"
ARG CONTAINER_TYPE="general"
RUN chmod +x /usr/local/bin/deep_learning_container.py \
  && sed -e "s/{{FRAMEWORK}}/${FRAMEWORK}/g" \
    -e "/{{FRAMEWORK_VERSION}}/d" \
    -e "s/{{CONTAINER_TYPE}}/${CONTAINER_TYPE}/g" \
    /tmp/bash_telemetry.sh.template >/usr/local/bin/bash_telemetry.sh \
  && chmod +x /usr/local/bin/bash_telemetry.sh \
  && rm /tmp/bash_telemetry.sh.template \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/etc/bash.bashrc \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/etc/bashrc \
  && echo 'source /usr/local/bin/bash_telemetry.sh' >>/root/.bashrc \
  && bash /tmp/setup_oss_compliance.sh python3 \
  && rm /tmp/setup_oss_compliance.sh \
  && rm -rf /var/lang/lib/python3.13/site-packages/pip \
    /var/lang/lib/python3.13/site-packages/pip-25.3.dist-info \
    /var/lang/lib/python3.13/ensurepip/_bundled
WORKDIR /var/task
ENTRYPOINT ["/lambda_entrypoint.sh", "python", "-m", "awslambdaric"]
CMD ["handler.handler"]