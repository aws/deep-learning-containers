# Changelog

Changelog for the Amazon Linux 2023-based vLLM images (`server-cuda`, `server-sagemaker-cuda`).

For Ubuntu-based image history, see [Release Notes](../../releasenotes/vllm/index.md).

* * *

## v1.3.0 — 2026-05-12

**Tags:** `server-cuda-v1.3` · `server-sagemaker-cuda-v1.3`

**vLLM source:** [3f5bd48](https://github.com/vllm-project/vllm/commit/3f5bd482f5c1a5dbdffbbf68d624e20bb7032013)
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

**vLLM source:** [8a8c9b5](https://github.com/vllm-project/vllm/commit/8a8c9b564ef015c76cf398200b8f0891e6e51bb8)
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

**vLLM source:** [6ef1efd5](https://github.com/vllm-project/vllm/commit/6ef1efd51f11106fc44deb9e7b2f5cd1247fc37e)
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

**vLLM source:** [6ef1efd5](https://github.com/vllm-project/vllm/commit/6ef1efd51f11106fc44deb9e7b2f5cd1247fc37e)

### Highlights

- Initial release of vLLM Server containers on Amazon Linux 2023
- Simplified tag format: `server-cuda[-vMAJOR[.MINOR[.PATCH]]]`
- OpenAI-compatible API server on port 8000
- Multi-GPU inference via tensor parallelism with NCCL
- EFA support for multi-node deployments
