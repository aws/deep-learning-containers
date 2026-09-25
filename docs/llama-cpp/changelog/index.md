# Changelog

Changelog for the Amazon Linux 2023-based llama.cpp images.

* * *

## llama.cpp 1.0.0 (v0.4.1), 2026-09-25

**Tags (x86, `llama-cpp`):** `server-cpu-v1` · `server-cuda-v1` · `server-sagemaker-cpu-v1` · `server-sagemaker-cuda-v1`

**Tags (ARM64, `llama-cpp-arm64`):** `server-cpu-v1` · `server-sagemaker-cpu-v1`

**llama.cpp source:** [v0.4.1](https://github.com/ggml-org/llama.cpp/releases/tag/v0.4.1) (build b10964)

### Highlights

- Upgraded llama.cpp from build b10433 to the stable release v0.4.1. Images now track the upstream stable (`v*`) release line instead of nightly `b*`
  builds.
- New model architectures: Maple 20B-A1B, Tencent Hy 4 (preview), and Spark2.5.
- `llama-server` fixes: an LRU hang on concurrent requests for the same model, speculative decoding after image input, and context checkpoint eviction
  on short prompts.
- `llama-server` now enables `--reasoning-preserve` by default.
- Upstream removed the deprecated `--mmap`, `--mlock`, and `--direct-io` flags. Use `--load-mode` instead. If you pass these flags through
  `SM_LLAMA_CPP_*` variables or a custom command, update them before moving to this image.
- Structured JSONL logging is available through `--log-jsonl`.
- See the [upstream release notes](https://github.com/ggml-org/llama.cpp/releases/tag/v0.4.1) for the full list of changes.

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
