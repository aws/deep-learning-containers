# Ported from awslabs/llm-hosting-container tei/1.9.3/gpu. Adjustments: tei-src/ COPY paths (staged by pre_build.sh), dropped --mount=type=secret, added dlc_major_version label + trailing apt upgrade.
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 AS base-builder

ENV SCCACHE=0.10.0
ENV RUSTC_WRAPPER=/usr/local/bin/sccache
ENV PATH="/root/.cargo/bin:${PATH}"
# aligned with `cargo-chef` version in `lukemathwalker/cargo-chef:latest-rust-1.92-bookworm`
ENV CARGO_CHEF=0.1.73

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  curl \
  libssl-dev \
  pkg-config \
  && rm -rf /var/lib/apt/lists/*

# Donwload and configure sccache
RUN curl -fsSL https://github.com/mozilla/sccache/releases/download/v$SCCACHE/sccache-v$SCCACHE-x86_64-unknown-linux-musl.tar.gz | tar -xzv --strip-components=1 -C /usr/local/bin sccache-v$SCCACHE-x86_64-unknown-linux-musl/sccache \
  && chmod +x /usr/local/bin/sccache

COPY docker/tei/1.9.3/gpu/tei-src/rust-toolchain.toml rust-toolchain.toml
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
RUN cargo install cargo-chef --version $CARGO_CHEF --locked

FROM base-builder AS planner

WORKDIR /usr/src

COPY docker/tei/1.9.3/gpu/tei-src/backends backends
COPY docker/tei/1.9.3/gpu/tei-src/core core
COPY docker/tei/1.9.3/gpu/tei-src/router router
COPY docker/tei/1.9.3/gpu/tei-src/Cargo.toml ./
COPY docker/tei/1.9.3/gpu/tei-src/Cargo.lock ./

RUN cargo chef prepare --recipe-path recipe.json

FROM base-builder AS builder

# sccache specific variables
ARG SCCACHE_GHA_ENABLED

# Limit parallelism
ARG RAYON_NUM_THREADS=4
ARG CARGO_BUILD_JOBS
ARG CARGO_BUILD_INCREMENTAL

WORKDIR /usr/src

COPY --from=planner /usr/src/recipe.json recipe.json

RUN cargo chef cook --release --recipe-path recipe.json && sccache -s;

RUN CUDA_COMPUTE_CAP=75 cargo chef cook --release --features candle-cuda-turing --recipe-path recipe.json && sccache -s;

RUN CUDA_COMPUTE_CAP=80 cargo chef cook --release --features candle-cuda --recipe-path recipe.json && sccache -s;

RUN CUDA_COMPUTE_CAP=90 cargo chef cook --release --features candle-cuda --recipe-path recipe.json && sccache -s;

RUN CUDA_COMPUTE_CAP=100 cargo chef cook --release --features candle-cuda --recipe-path recipe.json && sccache -s;

RUN CUDA_COMPUTE_CAP=120 cargo chef cook --release --features candle-cuda --recipe-path recipe.json && sccache -s;

COPY docker/tei/1.9.3/gpu/tei-src/backends backends
COPY docker/tei/1.9.3/gpu/tei-src/core core
COPY docker/tei/1.9.3/gpu/tei-src/router router
COPY docker/tei/1.9.3/gpu/tei-src/Cargo.toml ./
COPY docker/tei/1.9.3/gpu/tei-src/Cargo.lock ./

RUN CUDA_COMPUTE_CAP=75 cargo build --release --bin text-embeddings-router -F candle-cuda-turing && sccache -s;

RUN mv /usr/src/target/release/text-embeddings-router /usr/src/target/release/text-embeddings-router-75

RUN CUDA_COMPUTE_CAP=80 cargo build --release --bin text-embeddings-router -F candle-cuda && sccache -s;

RUN mv /usr/src/target/release/text-embeddings-router /usr/src/target/release/text-embeddings-router-80

RUN CUDA_COMPUTE_CAP=90 cargo build --release --bin text-embeddings-router -F candle-cuda && sccache -s;

RUN mv /usr/src/target/release/text-embeddings-router /usr/src/target/release/text-embeddings-router-90

RUN CUDA_COMPUTE_CAP=100 cargo build --release --bin text-embeddings-router -F candle-cuda && sccache -s;

RUN mv /usr/src/target/release/text-embeddings-router /usr/src/target/release/text-embeddings-router-100

RUN CUDA_COMPUTE_CAP=120 cargo build --release --bin text-embeddings-router -F candle-cuda && sccache -s;

RUN mv /usr/src/target/release/text-embeddings-router /usr/src/target/release/text-embeddings-router-120

FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04 AS sagemaker

ARG FRAMEWORK
ARG FRAMEWORK_VERSION
ARG CACHE_REFRESH

LABEL dlc_major_version="1"

ENV HUGGINGFACE_HUB_CACHE=/opt/ml/model \
  PORT=8080 \
  USE_FLASH_ATTENTION=True \
  LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  ca-certificates \
  libssl-dev \
  curl \
  cuda-compat-12-9 \
  && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/src/target/release/text-embeddings-router-75 /usr/local/bin/text-embeddings-router-75
COPY --from=builder /usr/src/target/release/text-embeddings-router-80 /usr/local/bin/text-embeddings-router-80
COPY --from=builder /usr/src/target/release/text-embeddings-router-90 /usr/local/bin/text-embeddings-router-90
COPY --from=builder /usr/src/target/release/text-embeddings-router-100 /usr/local/bin/text-embeddings-router-100
COPY --from=builder /usr/src/target/release/text-embeddings-router-120 /usr/local/bin/text-embeddings-router-120

ENV HF_HUB_USER_AGENT_ORIGIN=aws:sagemaker:gpu-cuda:inference:tei

# Security refresh — bust cache to pull latest apt patches on each daily rebuild
RUN apt-get update && apt-get upgrade -y \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --chmod=775 docker/tei/1.9.3/gpu/entrypoint.sh entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]