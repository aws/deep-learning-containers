# Serving GGUF Models using llama.cpp DLC

Production-ready Docker images for serving quantized [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) models with
[llama.cpp](https://github.com/ggml-org/llama.cpp) on {{ aws }}. Built on Amazon Linux 2023 with ongoing security patching.

Run large language models efficiently on x86 CPUs, NVIDIA GPUs, or AWS Graviton (ARM64) through the upstream `llama-server` OpenAI-compatible API.

## Images

The images ship for three hardware targets — x86 CPU, x86 NVIDIA GPU (CUDA), and Graviton (ARM64, CPU) — each in an {{ ec2_short }} and an
{{ sagemaker }} flavor. x86 images live in the `llama-cpp` repository; Graviton images live in `llama-cpp-arm64`. Every image serves on port **8080**.

| Platform | Architecture | Device | Image |
| --- | --- | --- | --- |
| {{ ec2_short }} | x86_64 | CPU | `public.ecr.aws/deep-learning-containers/llama-cpp:server-cpu-v1` |
| {{ ec2_short }} | x86_64 | GPU (CUDA) | `public.ecr.aws/deep-learning-containers/llama-cpp:server-cuda-v1` |
| {{ ec2_short }} | ARM64 (Graviton) | CPU | `public.ecr.aws/deep-learning-containers/llama-cpp-arm64:server-cpu-v1` |
| {{ sagemaker }} | x86_64 | CPU | `public.ecr.aws/deep-learning-containers/llama-cpp:server-sagemaker-cpu-v1` |
| {{ sagemaker }} | x86_64 | GPU (CUDA) | `public.ecr.aws/deep-learning-containers/llama-cpp:server-sagemaker-cuda-v1` |
| {{ sagemaker }} | ARM64 (Graviton) | CPU | `public.ecr.aws/deep-learning-containers/llama-cpp-arm64:server-sagemaker-cpu-v1` |

All images are also available on the ECR Public Gallery ([llama-cpp](https://gallery.ecr.aws/deep-learning-containers/llama-cpp),
[llama-cpp-arm64](https://gallery.ecr.aws/deep-learning-containers/llama-cpp-arm64)). For private ECR URIs, see [Image Access](../get_started/index.md).

## What's Included

Each image is a from-source build of the upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) project (tag `b10433`):

- **`llama-server`** — the OpenAI-compatible HTTP inference server (the default entrypoint)
- **`llama-cli`** and **`llama-bench`** — the interactive CLI and the benchmarking tool, on `PATH` for one-off use
- **libcurl-enabled build** (`LLAMA_CURL=ON`) — load a model directly from a HuggingFace repo at startup
- **Portable CPU dispatch** — the x86 CPU image bundles every microarchitecture backend (SSE4.2 through AVX-512/AMX) and selects the fastest at
  runtime; the Graviton image is tuned for Neoverse-V1 and runs across Graviton3/4
- **CUDA 13.0.2 runtime** (GPU image only) with automatic `cuda-compat` for forward compatibility

## API Endpoints

`llama-server` exposes the upstream OpenAI-compatible API on port 8080:

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/chat/completions` | Chat completions (OpenAI-compatible) |
| `POST /v1/completions` | Text completions (OpenAI-compatible) |
| `POST /v1/embeddings` | Embeddings (OpenAI-compatible) |
| `GET /v1/models` | Advertise the served model id |
| `GET /health` | Readiness health check |
| `POST /invocations` | {{ sm_short }} alias → `/v1/chat/completions` |
| `GET /ping` | {{ sm_short }} readiness alias → `/health` |

On {{ sagemaker }} the container is fronted by nginx, which maps `GET /ping` to `/health` and `POST /invocations` to `/v1/chat/completions`; every
other path is proxied to `llama-server` unchanged, so the full `/v1/*` API stays reachable. See [EC2 Deployment](deployment/ec2.md) and
[{{ sagemaker }} Deployment](deployment/sagemaker.md) for examples, and [Configuration](configuration.md) for every launch option.

## How We Build

These images are curated builds tracking the [llama.cpp](https://github.com/ggml-org/llama.cpp) project:

- **Built from upstream releases** — images are built from a pinned llama.cpp build tag, each gated by our regression test suite before publication.
- **Regression-tested** — validated against quantized GGUF models on {{ ec2_short }} and {{ sagemaker }} on every release. See
  [Supported Models](models/index.md).
- **Security-patched** — continuously maintained with security patches from {{ aws }} on an Amazon Linux 2023 base.
