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
FROM nvidia/cuda:12.9.1-runtime-amzn2023 as builder-base
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN dnf install -y --setopt=install_weak_deps=False tar gzip findutils \
  && dnf clean all && rm -rf /var/cache/dnf

# ============================================================
# Shared builder base — devel (CUDA devel + uv + git)
# ============================================================
FROM nvidia/cuda:12.9.1-devel-amzn2023 as builder-base-devel
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN dnf install -y --setopt=install_weak_deps=False tar gzip git findutils \
  && dnf clean all && rm -rf /var/cache/dnf

# ============================================================
# FFmpeg builder (compiled from source with NVENC/NVDEC GPU acceleration)
# ============================================================
FROM nvidia/cuda:12.9.1-devel-amzn2023 as ffmpeg-builder
ENV PATH="/usr/local/cuda/bin:${PATH}"
RUN dnf install -y --setopt=install_weak_deps=False \
  tar xz gcc gcc-c++ make pkgconfig nasm yasm git diffutils \
  && dnf clean all && rm -rf /var/cache/dnf \
  && git clone --branch n12.2.72.0 --depth 1 https://github.com/FFmpeg/nv-codec-headers.git /tmp/nv-codec-headers \
  && cd /tmp/nv-codec-headers && make install && cd /tmp && rm -rf nv-codec-headers \
  && curl -fsSL https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n8.0.1.tar.gz | tar xz -C /tmp \
  && mv /tmp/FFmpeg-n8.0.1 /tmp/ffmpeg \
  && cd /tmp/ffmpeg \
  && PKG_CONFIG_PATH="/usr/local/lib/pkgconfig" ./configure \
    --prefix=/usr/local \
    --disable-static \
    --enable-shared \
    --disable-doc \
    --disable-htmlpages \
    --disable-manpages \
    --disable-podpages \
    --disable-txtpages \
    --disable-debug \
    --enable-nonfree \
    --enable-cuda-nvcc \
    --enable-libnpp \
    --extra-cflags=-I/usr/local/cuda/include \
    --extra-ldflags=-L/usr/local/cuda/lib64 \
    --nvccflags="-gencode arch=compute_75,code=sm_75 -O2" \
  && make -j$(nproc) \
  && make install \
  && strip /usr/local/bin/ffmpeg /usr/local/bin/ffprobe \
  && rm -rf /tmp/ffmpeg

# ============================================================
# Base builder (Python 3.13 — awslambdaric, requests, pip-licenses)
# ============================================================
FROM builder-base as builder-base-py3
COPY --from=lambda-python /var/lang/bin /var/lang/bin
COPY --from=lambda-python /var/lang/include /var/lang/include
COPY --from=lambda-python /var/lang/lib /var/lang/lib
RUN rm -rf /var/lang/lib/python3.13/site-packages
ENV PATH="/var/lang/bin:$PATH" \
  UV_PROJECT_ENVIRONMENT="/var/lang"
RUN printf 'home = /var/lang/bin\nimplementation = CPython\n' >/var/lang/pyvenv.cfg
COPY ./docker/lambda/base/pyproject.toml ./docker/lambda/base/uv.lock /tmp/build/
WORKDIR /tmp/build
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-dev --no-install-project \
  && (find /var/lang -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true) \
  && find /var/lang -type f -name "*.pyc" -delete \
  && find /var/lang -type f -name "*.pyo" -delete

# ============================================================
# CuPy builder (Python 3.13 — awslambdaric, CuPy, NumPy, SciPy, etc.)
# ============================================================
FROM builder-base as builder-cupy-py3
COPY --from=lambda-python /var/lang/bin /var/lang/bin
COPY --from=lambda-python /var/lang/include /var/lang/include
COPY --from=lambda-python /var/lang/lib /var/lang/lib
RUN rm -rf /var/lang/lib/python3.13/site-packages
ENV PATH="/var/lang/bin:$PATH" \
  UV_PROJECT_ENVIRONMENT="/var/lang"
RUN printf 'home = /var/lang/bin\nimplementation = CPython\n' >/var/lang/pyvenv.cfg
COPY ./docker/lambda/cupy/pyproject.toml ./docker/lambda/cupy/uv.lock /tmp/build/
WORKDIR /tmp/build
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-dev --no-install-project \
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
ENV PATH="/var/lang/bin:$PATH" \
  UV_PROJECT_ENVIRONMENT="/var/lang"
RUN printf 'home = /var/lang/bin\nimplementation = CPython\n' >/var/lang/pyvenv.cfg
COPY ./docker/lambda/pytorch/pyproject.toml ./docker/lambda/pytorch/uv.lock /tmp/build/
WORKDIR /tmp/build
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-dev --no-install-project \
  && (find /var/lang -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true) \
  && (find /var/lang -type d -name tests -exec rm -rf {} + 2>/dev/null; true) \
  && find /var/lang -type f -name "*.pyc" -delete \
  && find /var/lang -type f -name "*.pyo" -delete

# ============================================================
# OSS compliance builders (isolated — prevents dependency closure modification)
# ============================================================
FROM nvidia/cuda:12.9.1-runtime-amzn2023 as oss-base-py3
COPY --from=builder-base-py3 /var/lang /var/lang
COPY ./scripts/common/setup_oss_compliance.sh /tmp/setup_oss_compliance.sh
RUN PATH="/var/lang/bin:$PATH" bash /tmp/setup_oss_compliance.sh python3 \
  && touch /root/THIRD_PARTY_SOURCE_CODE_URLS

FROM nvidia/cuda:12.9.1-runtime-amzn2023 as oss-cupy-py3
COPY --from=builder-cupy-py3 /var/lang /var/lang
COPY ./scripts/common/setup_oss_compliance.sh /tmp/setup_oss_compliance.sh
RUN PATH="/var/lang/bin:$PATH" bash /tmp/setup_oss_compliance.sh python3 \
  && touch /root/THIRD_PARTY_SOURCE_CODE_URLS

FROM nvidia/cuda:12.9.1-runtime-amzn2023 as oss-pytorch-py3
COPY --from=builder-pytorch-py3 /var/lang /var/lang
COPY ./scripts/common/setup_oss_compliance.sh /tmp/setup_oss_compliance.sh
RUN PATH="/var/lang/bin:$PATH" bash /tmp/setup_oss_compliance.sh python3 \
  && printf '\n** FFmpeg; version n8.0.1 -- https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n8.0.1.tar.gz\n' >> /root/BUILD_FROM_SOURCE_PACKAGES_LICENCES \
  && printf '\nFFmpeg n8.0.1 https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n8.0.1.tar.gz\n' >> /root/THIRD_PARTY_SOURCE_CODE_URLS \
  && printf '\n** FFprobe; version n8.0.1 -- https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n8.0.1.tar.gz\n' >> /root/BUILD_FROM_SOURCE_PACKAGES_LICENCES \
  && printf '\nFFprobe n8.0.1 https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n8.0.1.tar.gz\n' >> /root/THIRD_PARTY_SOURCE_CODE_URLS

# ============================================================
# Runtime: Base Python 3.13 (CUDA + Python + RIC + RIE)
# ============================================================
FROM nvidia/cuda:12.9.1-runtime-amzn2023 as base-py3
LABEL maintainer="Amazon AI"
LABEL dlc_major_version="1"
LABEL dlc_minor_version="0"
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
  && rm -rf /var/lang/lib/python3.13/ensurepip/_bundled
COPY --from=oss-base-py3 /root/THIRD_PARTY_SOURCE_CODE_URLS /root/THIRD_PARTY_SOURCE_CODE_URLS
COPY --from=oss-base-py3 /root/PYTHON_PACKAGES_LICENSES /root/PYTHON_PACKAGES_LICENSES
COPY --from=oss-base-py3 /root/LINUX_PACKAGES_LICENSES /root/LINUX_PACKAGES_LICENSES
COPY --from=oss-base-py3 /root/BUILD_FROM_SOURCE_PACKAGES_LICENCES /root/BUILD_FROM_SOURCE_PACKAGES_LICENCES
COPY --from=oss-base-py3 /usr/local/bin/testOSSCompliance /usr/local/bin/testOSSCompliance
WORKDIR /var/task
ENTRYPOINT ["/lambda_entrypoint.sh", "python", "-m", "awslambdaric"]
CMD ["handler.handler"]

# ============================================================
# Runtime: CuPy Python 3.13 (base + CuPy, NumPy, SciPy)
# ============================================================
FROM nvidia/cuda:12.9.1-runtime-amzn2023 as cupy-py3
LABEL maintainer="Amazon AI"
LABEL dlc_major_version="1"
LABEL dlc_minor_version="0"
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
  && rm -rf /var/lang/lib/python3.13/ensurepip/_bundled
COPY --from=oss-cupy-py3 /root/THIRD_PARTY_SOURCE_CODE_URLS /root/THIRD_PARTY_SOURCE_CODE_URLS
COPY --from=oss-cupy-py3 /root/PYTHON_PACKAGES_LICENSES /root/PYTHON_PACKAGES_LICENSES
COPY --from=oss-cupy-py3 /root/LINUX_PACKAGES_LICENSES /root/LINUX_PACKAGES_LICENSES
COPY --from=oss-cupy-py3 /root/BUILD_FROM_SOURCE_PACKAGES_LICENCES /root/BUILD_FROM_SOURCE_PACKAGES_LICENCES
COPY --from=oss-cupy-py3 /usr/local/bin/testOSSCompliance /usr/local/bin/testOSSCompliance
WORKDIR /var/task
ENTRYPOINT ["/lambda_entrypoint.sh", "python", "-m", "awslambdaric"]
CMD ["handler.handler"]

# ============================================================
# Runtime: PyTorch Python 3.13 (base + PyTorch, FFmpeg, SAM2)
# ============================================================
FROM nvidia/cuda:12.9.1-runtime-amzn2023 as pytorch-py3
LABEL maintainer="Amazon AI"
LABEL dlc_major_version="1"
LABEL dlc_minor_version="0"
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
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
  && rm -rf /var/lang/lib/python3.13/ensurepip/_bundled
COPY --from=oss-pytorch-py3 /root/THIRD_PARTY_SOURCE_CODE_URLS /root/THIRD_PARTY_SOURCE_CODE_URLS
COPY --from=oss-pytorch-py3 /root/PYTHON_PACKAGES_LICENSES /root/PYTHON_PACKAGES_LICENSES
COPY --from=oss-pytorch-py3 /root/LINUX_PACKAGES_LICENSES /root/LINUX_PACKAGES_LICENSES
COPY --from=oss-pytorch-py3 /root/BUILD_FROM_SOURCE_PACKAGES_LICENCES /root/BUILD_FROM_SOURCE_PACKAGES_LICENCES
COPY --from=oss-pytorch-py3 /usr/local/bin/testOSSCompliance /usr/local/bin/testOSSCompliance
WORKDIR /var/task
ENTRYPOINT ["/lambda_entrypoint.sh", "python", "-m", "awslambdaric"]
CMD ["handler.handler"]