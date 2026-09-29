#!/usr/bin/env bash
# Print the ARG lines for docker/lambda/Dockerfile's pinned dependencies.
#
# Run after bumping any version below, then paste the output over the matching
# ARG lines. Versions are read from the Dockerfile, so bump there first.
#
# Usage: bash scripts/ci/checks/refresh_pins.sh [dockerfile]
set -euo pipefail

DF="${1:-docker/lambda/Dockerfile}"
arg() { grep -m1 "^ARG $1=" "$DF" | cut -d= -f2-; }

CUDA=$(arg CUDA_VERSION)
PY=$(arg PYTHON_VERSION)
UV=$(arg UV_VERSION)
RIE=$(arg AWS_LAMBDA_RIE_VERSION)
FFMPEG=$(arg FFMPEG_VERSION)
NVCH=$(arg NV_CODEC_HEADERS_VERSION)
SCCACHE=$(arg SCCACHE_VERSION)
RUSTUP=$(arg RUSTUP_VERSION)

digest() { docker buildx imagetools inspect "$1" | awk '/^Digest:/{print $2; exit}'; }
remote_sha() { curl -fsSL "$1" | sha256sum | cut -d' ' -f1; }

echo "ARG LAMBDA_PYTHON_DIGEST=$(digest "public.ecr.aws/lambda/python:${PY}")"
echo "ARG CUDA_RUNTIME_DIGEST=$(digest "nvidia/cuda:${CUDA}-runtime-amzn2023")"
echo "ARG CUDA_DEVEL_DIGEST=$(digest "nvidia/cuda:${CUDA}-devel-amzn2023")"
echo "ARG UV_DIGEST=$(digest "ghcr.io/astral-sh/uv:${UV}")"
echo "ARG AWS_LAMBDA_RIE_SHA256=$(remote_sha "https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/download/v${RIE}/aws-lambda-rie-x86_64")"
echo "ARG FFMPEG_SHA256=$(remote_sha "https://ffmpeg.org/releases/ffmpeg-${FFMPEG}.tar.xz")"
echo "ARG NV_CODEC_HEADERS_COMMIT=$(git ls-remote https://github.com/FFmpeg/nv-codec-headers.git "refs/tags/n${NVCH}" | cut -f1)"
echo "ARG SCCACHE_SHA256=$(curl -fsSL "https://github.com/mozilla/sccache/releases/download/v${SCCACHE}/sccache-v${SCCACHE}-x86_64-unknown-linux-musl.tar.gz.sha256")"
echo "ARG RUSTUP_SHA256=$(curl -fsSL "https://static.rust-lang.org/rustup/archive/${RUSTUP}/x86_64-unknown-linux-gnu/rustup-init.sha256" | cut -d' ' -f1)"

# FFMPEG_GPG_KEY changes only if FFmpeg rotates its release key; verify at
# ffmpeg.org/download.html before changing it, and re-export the vendored
# docker/lambda/keys/ffmpeg-release.asc to match.
