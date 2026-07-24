# Changelog

Changelog for the Amazon Linux 2023-based vLLM images (`server-cuda`, `server-sagemaker-cuda`).

* * *

## v2.1.0 — 2026-07-02

**Tags:** `server-cuda-v2.1` · `server-sagemaker-cuda-v2.1`

**vLLM source:** [7b3d595](https://github.com/vllm-project/vllm/commit/7b3d595eb197d714052ce296cc8b124f0dc8af31) (`0.24.0+amzn2023.7b3d595e`)

**Bundled versions:** CUDA 13.0.2 · Python 3.12 · FlashInfer 0.6.12 · DeepEP [73b6ea4](https://github.com/deepseek-ai/DeepEP/commit/73b6ea4)

### Highlights

- **vLLM 0.24.0** — minor version bump from 0.22.1rc0 (v2.0)
- **FlashInfer 0.6.12** — upgraded from 0.6.11.post2
- **transformers `<5.10` pin removed** — vLLM 0.24.0 requires transformers ≥ 5.5.3, so the previous pin is dropped
- **`mistral_common` now an optional import** — audio dependencies (`mistral_common[audio]`, av, scipy, soundfile) are installed explicitly for
  Voxtral / ASR serving

### New Model Support

- Mellum2-12B-A2.5B-Thinking (`MellumForCausalLM`)

### Notes

- GPU device selection refactor upstream: vLLM no longer sets `CUDA_VISIBLE_DEVICES` internally and adds a `--device-ids` flag. Single-server
  tensor-parallel serving is unaffected.
- Models removed upstream (ERNIE, Xverse, Dots1, Bamba, Mono-InternVL) are not part of the DLC test matrix.

* * *

## v2.0.0 — 2026-06-05

**Tags:** `server-cuda-v2.0` · `server-sagemaker-cuda-v2.0`

**vLLM source:** [6aabe22](https://github.com/vllm-project/vllm/commit/6aabe221a56052965e6bb0a95e9ec682d046a6e7) (`0.22.1rc0+amzn2023.6aabe221`)

**Bundled versions:** CUDA 13.0.2 · Python 3.12 · FlashInfer 0.6.11.post2 · DeepEP [73b6ea4](https://github.com/deepseek-ai/DeepEP/commit/73b6ea4)

### Highlights

- **vLLM 0.22.1rc0** — major version bump from 0.20.0.dev361 (v1.4)
- **CUDA 13.0.2** — upgraded from 12.9.1; requires NVIDIA driver 580+
- **FlashInfer 0.6.11.post2** — upgraded from 0.6.8.post1; precompiled cubins now bundled
- **EC2 entrypoint simplified** — uses `vllm serve` CLI instead of `python3 -m vllm.entrypoints.openai.api_server`
- **nixl-cu13 fix** — KV connector NIXL now correctly linked against CUDA 13
- **transformers pinned to <5.10** — avoids AttributeError on Voxtral with mistral-common 1.11.2

### New Model Support

- Qwen3-Embedding-0.6B and Qwen3-VL-Embedding-2B (embedding)
- Qwen3-Reranker-4B (reranking)
- IBM Granite-Speech-4.1-2B (ASR)
- Gemma 4 family: 26B-A4B-it, 31B-it, E4B-it, E2B-it
- Qwen3.5 (0.8B, 2B) and Qwen3.6 (27B, 35B-A3B)

### Security

- CVE-2025-33219: explicit `cuda-compat-13-0` upgrade in EC2 and SageMaker stages
- `model-hosting-container-standards` bumped to ≥0.1.15

* * *

## v1.4.0 — 2026-05-22

**Tags:** `server-cuda-v1.4` · `server-sagemaker-cuda-v1.4`

**vLLM source:** [3f5bd48](https://github.com/vllm-project/vllm/commit/3f5bd482f5c1a5dbdffbbf68d624e20bb7032013) (`0.20.0.dev361+amzn2023.3f5bd482`)

**Bundled versions:** CUDA 12.9.1 · Python 3.12 · FlashInfer 0.6.8.post1 · DeepEP [73b6ea4](https://github.com/deepseek-ai/DeepEP/commit/73b6ea4)

### Highlights

- **SageMaker route middleware** ([#6096](https://github.com/aws/deep-learning-containers/pull/6096)) — `/invocations` requests can now be routed to
  any vLLM endpoint via `X-Amzn-SageMaker-Custom-Attributes: route=<path>`
- **libsndfile** added to the SageMaker image for audio I/O in network-isolated deployments

### New Model Support

- **Voxtral-Mini-4B-Realtime-2602** — Mistral audio transcription via `/v1/audio/transcriptions`

* * *

## v1.3.0 — 2026-05-12

**Tags:** `server-cuda-v1.3` · `server-sagemaker-cuda-v1.3`

**vLLM source:** [3f5bd48](https://github.com/vllm-project/vllm/commit/3f5bd482f5c1a5dbdffbbf68d624e20bb7032013) (`0.20.0.dev361+amzn2023.3f5bd482`)

**Bundled versions:** CUDA 12.9.1 · Python 3.12 · FlashInfer 0.6.8.post1 · DeepEP commit 73b6ea4

### Highlights

- **SageMaker standard-supervisor integration** — process auto-recovery on crash, dynamic dependency installation from `requirements.txt` in model
  artifacts, and custom handler support via `model.py`
- **Gemma 4 fixes** — pipeline parallelism, MoE weight loading, CUDA graph capture, multimodal memory, tool calling stability, MTP speculative
  decoding support
- **DeepSeek V4 fixes** — numerical correctness for topk, tool calling for non-streaming, disaggregated P/D serving, performance optimizations

### SageMaker Features (new)

- Process supervision with auto-recovery (configurable via `PROCESS_AUTO_RECOVERY`)
- Dynamic `requirements.txt` installation before server startup
- Custom `/ping` and `/invocations` handler support via `model.py` in model artifacts
- LoRA adapter routing via request headers

### Model Fixes

- Gemma 4: fix PP, MoE expert weight remapping, activation mismatch, infinite loop in tool parser, chat template sync
- Gemma 4: add MTP speculative decoding support
- DeepSeek V4: fix topk numerical issue, repeated RoPE cache initialization, disaggregated serving
- DeepSeek V4: integrate tile kernel head_compute_mix_kernel for improved performance

* * *

## v1.2.0 — 2026-04-30

**Tags:** `server-cuda-v1.2` · `server-sagemaker-cuda-v1.2`

**vLLM source:** [8a8c9b5](https://github.com/vllm-project/vllm/commit/8a8c9b564ef015c76cf398200b8f0891e6e51bb8) (`0.20.0.dev60+amzn2023.8a8c9b56`)

**Bundled versions:** CUDA 12.9.1 · Python 3.12 · FlashInfer 0.6.8.post1 · DeepEP commit 73b6ea4

### Highlights

- **DeepSeek V4 support** — full model support including Pro and Flash variants, multi-stream pre-attention GEMM, MLA + group FP8 fusion
- **Qwen3.5 / Qwen3.6 / Qwen3-Coder fixes** — LoRA for MoE, double gate call fix, tool calling fix
- **Gemma 4 fixes** — multimodal embedder norm order, bidirectional vision attention for sliding layers
- Removed vLLM RayServe setup (deprecated)
- Fixed telemetry script version matching with proper PEP 440 compatibility

### New Model Support

- DeepSeek V4 Pro and Flash
- DeepSeek V4 base model

### Model Fixes

- DeepSeek V4: token leakage fix, inductor error fix, KV block release for skipped P-ranks with MLA
- Qwen3.5: LoRA support for MoE, double gate call fix
- Qwen3: tool calling fix for `<tool_call>` as implicit reasoning end
- Gemma 4: multimodal embedder norm order fix, bidirectional vision attention

* * *

## v1.1.0 — 2026-04-28

**Tags:** `server-cuda-v1.1` · `server-sagemaker-cuda-v1.1`

**vLLM source:** [6ef1efd5](https://github.com/vllm-project/vllm/commit/6ef1efd51f11106fc44deb9e7b2f5cd1247fc37e) (`0.19.1+amzn2023.6ef1efd5`)

**Bundled versions:** CUDA 12.9.1 · Python 3.12 · FlashInfer 0.6.7 · LMCache 0.4.5.dev0 (custom build)

### Highlights

- **LMCache bidirectional NIXL cache probe** — enables disaggregated prefill/decode (P/D) deployments with bidirectional cache querying between
  prefill and decode workers

### Changes

- Override LMCache with source build from commit [7f60057](https://github.com/LMCache/LMCache/commit/7f60057ce37102bf7e7a519901930d6b9a874136) for
  bidirectional NIXL feature
- LMCache version: 0.4.5.dev0+amzn2023.7f60057c

* * *

## v1.0.0 — 2026-04-25

**Tags:** `server-cuda-v1.0` · `server-sagemaker-cuda-v1.0`

**vLLM source:** [6ef1efd5](https://github.com/vllm-project/vllm/commit/6ef1efd51f11106fc44deb9e7b2f5cd1247fc37e) (`0.19.1+amzn2023.6ef1efd5`)

**Bundled versions:** CUDA 12.9.1 · Python 3.12 · FlashInfer 0.6.7

### Highlights

- Initial release of vLLM Server containers on Amazon Linux 2023
- Simplified tag format: `server-cuda[-vMAJOR[.MINOR[.PATCH]]]`
- OpenAI-compatible API server on port 8000
- Multi-GPU inference via tensor parallelism with NCCL
- EFA support for multi-node deployments
