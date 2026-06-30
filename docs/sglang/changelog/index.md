# Changelog

Changelog for the Amazon Linux 2023-based SGLang images (`server-cuda`, `server-sagemaker-cuda`).

* * *

## v1.1.0 — 2026-06-29

**Tags:** `server-cuda-v1.1` · `server-sagemaker-cuda-v1.1`

**SGLang source:** [66ab5c9](https://github.com/sgl-project/sglang/commit/66ab5c9c7a425ef746725b8a8d7210d434eadac2) (`0.5.13+amzn2023.66ab5c9`)

**Bundled versions:** CUDA 13.0.3 · Python 3.12 · PyTorch 2.11.0 · sgl-kernel 0.4.3 · FlashInfer 0.6.11.post1 · Mooncake 0.3.9 · NCCL 2.28.3

### Highlights

- Bumped SGLang to `0.5.13` (upstream commit `66ab5c9`)
- Added **[NIXL](https://github.com/ai-dynamo/nixl)** KV connector (`nixl` + matching `nixl-cu13`) for prefill/decode disaggregation KV transfer
- Added **[runai-model-streamer](https://github.com/run-ai/runai-model-streamer)** with the `[s3,gcs,azure]` extras for fast weight streaming from
  object storage (`sglang[all]` omits the `[s3]` extra)

### Security

- Patched starlette CVE [GHSA-82w8-qh3p-5jfq](https://github.com/advisories/GHSA-82w8-qh3p-5jfq) — pinned `starlette>=1.3.1`
- Removed the build-only `rust/` source tree from the runtime stage (only the compiled `sglang-router` binary ships); resolves a pyo3 CVE
  [GHSA-36hh-v3qg-5jq4](https://github.com/advisories/GHSA-36hh-v3qg-5jq4) flagged on the leftover `Cargo.lock`

* * *

## v1.0.0 — 2026-06-12

**Tags:** `server-cuda-v1.0` · `server-sagemaker-cuda-v1.0`

**SGLang source:** [578d27e](https://github.com/sgl-project/sglang/commit/578d27e56acd6786eb7067b9e1583232fd1d998e) (`0.5.12+amzn2023.578d27e`)

**Bundled versions:** CUDA 13.0.3 · Python 3.12 · PyTorch 2.11.0 · sgl-kernel 0.4.3 · FlashInfer 0.6.11.post1 · Mooncake 0.3.9 · NCCL 2.28.3

### Highlights

- Initial release of SGLang Server containers on Amazon Linux 2023
- Built from upstream SGLang source (not the pre-built `lmsysorg/sglang` image)
- Simplified tag format: `server-cuda[-vMAJOR[.MINOR[.PATCH]]]`
- OpenAI-compatible API server on port 30000 (EC2 / EKS) and 8080 (SageMaker)
- Multi-GPU inference via tensor parallelism with NCCL
- CUDA 13.0 build targeting H100 (sm_90) and Blackwell (sm_100, sm_103)
- EFA support for multi-node deployments
- DeepEP expert-parallel kernels and Mooncake KV-cache transfer bundled for large MoE models
