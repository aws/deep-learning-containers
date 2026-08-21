# Changelog

Changelog for the Amazon Linux 2023-based llama.cpp images.

* * *

## llama.cpp 1.0.0 (b10433) — 2026-08-21

**Tags (x86, `llama-cpp`):** `server-cpu-v1` · `server-cuda-v1` · `server-sagemaker-cpu-v1` · `server-sagemaker-cuda-v1`

**Tags (ARM64, `llama-cpp-arm64`):** `server-cpu-v1` · `server-sagemaker-cpu-v1`

**llama.cpp source:** [b10433](https://github.com/ggml-org/llama.cpp/releases/tag/b10433)

### Highlights

- Initial release of llama.cpp inference containers on Amazon Linux 2023.
- Serves quantized GGUF models through the upstream `llama-server` OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`,
  `/v1/models`).
- Three hardware targets: **x86 CPU**, **x86 NVIDIA GPU (CUDA 13.0.2)**, and **AWS Graviton (ARM64) CPU** — each in an {{ ec2_short }} and an
  {{ sagemaker }} flavor (port 8080).
- The x86 CPU image bundles every microarchitecture backend (SSE4.2 → AVX-512/AMX) with runtime dispatch, and the Graviton image is tuned for
  Neoverse-V1.
- {{ sagemaker }} images front `llama-server` with nginx (`/ping` → `/health`, `/invocations` → `/v1/chat/completions`) and are configured via
  `SM_LLAMA_CPP_*` environment variables.
- Built from upstream llama.cpp with `LLAMA_CURL=ON` for direct HuggingFace model downloads. `llama-cli` and `llama-bench` are included too.
- Built on Python 3.12.
