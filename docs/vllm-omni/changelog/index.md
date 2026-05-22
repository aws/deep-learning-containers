# Changelog

Changelog for the Amazon Linux 2023-based vLLM-Omni images (`omni-cuda`, `omni-sagemaker-cuda`).

* * *

## v1.3.0 — 2026-05-21

**Tags:** `omni-cuda-v1.3` · `omni-sagemaker-cuda-v1.3`

**vLLM-Omni source:** [v0.21.0rc1](https://github.com/vllm-project/vllm-omni/releases/tag/v0.21.0rc1) (pre-release, tracking upstream vLLM v0.21.0)

### Highlights

- Upgraded to vLLM-Omni 0.21.0rc1, aligned with upstream vLLM v0.21.0
- Cherry-picked upstream Dockerfile fixes for cublas headers (JIT), flashinfer cubin layering, and the `nixl-cu13` install ordering for matching
  `nixl_ep_cpp.so`

### Fixes

- **Voice-clone TTS (Qwen3-TTS-Base) throughput restored** — the upstream Code2Wav decode-chunk un-batching regression flagged in v1.1 is resolved in
  vllm-omni 0.21.0rc1.

### Known Issues

- **Transformers pinned to `<5.9.0`.** Transformers 5.9.0 removed the deprecated `input_embeds` alias and the `cache_position` kwarg from
  `create_causal_mask` / `create_sliding_window_causal_mask`, which breaks Qwen3-TTS decode in vllm-omni 0.21.0rc1. Pin will be dropped once a
  vllm-omni release containing the upstream fix ships.

* * *

## v1.2.0 — 2026-05-18

**Tags:** `omni-cuda-v1.2` · `omni-sagemaker-cuda-v1.2`

**vLLM-Omni source:** [v0.20.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.20.0) (unchanged from v1.1)

### Changes

- **SageMaker `/v1/videos` and `/v1/videos/sync` now require `multipart/form-data` directly.** The routing middleware no longer auto-converts JSON
  request bodies to multipart. Clients must build the multipart body locally and pass `ContentType="multipart/form-data; boundary=..."` to
  `InvokeEndpoint`; SageMaker forwards the body and `ContentType` through to the model server unchanged.
- See `examples/vllm-omni/sagemaker/deploy_video_sync.py` for the updated invocation pattern.

### Migration

- Clients that previously sent JSON to `/v1/videos*` via SageMaker `CustomAttributes` routing must switch to a pre-built multipart body. JSON requests
  to these routes will now reach the model server unconverted and fail.

* * *

## v1.1.0 — 2026-05-12

**Tags:** `omni-cuda-v1.1` · `omni-sagemaker-cuda-v1.1`

**vLLM-Omni source:** [v0.20.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.20.0)

### Highlights

- Upgraded to vLLM-Omni 0.20.0, aligned with upstream vLLM v0.20.0
- CUDA bumped from 12.9 to 13.0
- New `/v1/audio/generate` endpoint for diffusion-based audio generation
- New `/v1/videos/sync` endpoint — blocks until complete and returns raw MP4, enabling video generation on SageMaker

### New Models

- **ERNIE-Image-Turbo** — 8-step distilled DiT image generation
- **Wan2.1-VACE-1.3B** — unified video creation/editing pipeline
- **Stable-Audio-Open-1.0** — text-to-audio diffusion
- **CosyVoice3-0.5B** — zero-shot voice cloning

### Changes

- Added `numactl` for fastsafetensors compatibility with CUDA 13
- Added `VLLM_ENABLE_CUDA_COMPATIBILITY=0` env (set to `1` for hosts with older NVIDIA drivers)
- Removed `sox` system dependency (no longer needed by vllm-omni)
- Expanded smoke-test matrix from 6 to 9 models with performance benchmarks

### Known Issues

- Voice-clone TTS (Qwen3-TTS-Base) throughput regression vs v1.0 due to upstream Code2Wav un-batching. Fix merged upstream, pending next release.

* * *

## v1.0.0 — 2026-04-24

**Tags:** `omni-cuda-v1.0` · `omni-sagemaker-cuda-v1.0`

**vLLM-Omni source:** [v0.18.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.18.0)

### Highlights

- Initial release of vLLM-Omni containers on Amazon Linux 2023
- Serves TTS, image generation, video generation, and multimodal chat through OpenAI-compatible APIs
- SageMaker routing middleware for dispatching `/invocations` to any omni endpoint via `CustomAttributes`
- Built on CUDA 12.9 with Python 3.12

### Supported Models at Launch

- Qwen3-TTS-1.7B (preset voice + voice-clone)
- FLUX.2-klein-4B (image generation)
- Wan2.1-T2V-1.3B (video generation)
- Qwen2.5-Omni-3B (multimodal chat)
