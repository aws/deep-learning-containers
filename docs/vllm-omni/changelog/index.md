# Changelog

Changelog for the Amazon Linux 2023-based vLLM-Omni images (`omni-cuda`, `omni-sagemaker-cuda`).

* * *

## v1.5.0 — 2026-08-07

**Tags:** `omni-cuda-v1.5` · `omni-sagemaker-cuda-v1.5`

**vLLM-Omni source:** [v0.26.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.26.0)

### Highlights

- Upgraded to vLLM-Omni 0.26.0 — the first stable release since 0.20.0, skipping the `0.21.0rc1` pre-release the DLC had been tracking — aligned with
  upstream vLLM v0.26.0.
- **vLLM v0.26.0 Rust frontend (`vllm-rs`).** The build now compiles upstream's new Rust extension (rustup + protoc + `build_rust.sh`) before the
  wheel build; `VLLM_REQUIRE_RUST_FRONTEND=1` guards against silently shipping a wheel missing the `_rust_*.so` artifacts.
- **Bidirectional WebSocket streaming on SageMaker.** The SageMaker image now advertises
  `com.amazonaws.sagemaker.capabilities.bidirectional-streaming=true` and bridges SageMaker's Bidirectional Streaming API to vLLM-Omni's native
  WebSocket routes (e.g. `/v1/audio/speech/stream`) for low-latency streaming TTS and realtime sessions. See
  [SageMaker Deployment](../deployment/sagemaker.md).
- **CosyVoice3 fixed on 0.26.0.** `s3tokenizer==0.3.0` is now bundled — CosyVoice3's model code hard-imports it at load in 0.26.0, but upstream ships
  it only in a dev extra, so the base install previously failed with `ModuleNotFoundError`.
- FlashInfer bumped to 0.6.14; the local `transformers <5.9.0` cap is dropped (0.26.0 handles the Qwen3-TTS breakage in code).

### Changes

- FlashInfer JIT-cache install switched from `--extra-index-url` to `--index-url` (flashinfer-cubin left PyPI as of 0.6.14).
- Dropped the `+PTX` suffix from `torch_cuda_arch_list` (upstream filters the global PTX flag as a no-op).

### Known Issues

- The DeepGEMM `_C` extension is not built for this bump; the image falls back to the Python path. Follow-up planned.

* * *

## v1.4.0 — 2026-07-02

**Tags:** `omni-cuda-v1.4` · `omni-sagemaker-cuda-v1.4`

**vLLM-Omni source:** [v0.21.0rc1](https://github.com/vllm-project/vllm-omni/releases/tag/v0.21.0rc1) (unchanged from v1.3)

**DLC PR:** [#6298](https://github.com/aws/deep-learning-containers/pull/6298)

### Changes

- **SageMaker `/v1/videos` and `/v1/videos/sync` now accept `application/json` again.** The routing middleware restores JSON→multipart conversion for
  the form-data video routes, so clients can send a plain JSON body instead of hand-building multipart. The existing `multipart/form-data` path is
  unchanged (byte-for-byte passthrough), so callers already sending multipart need no changes.

### Notes

- No framework bump — still tracks vLLM-Omni 0.21.0rc1 (upstream vLLM v0.21.0). This is a DLC-minor release (v1.3 → v1.4) scoped to the SageMaker
  video-route change above.

* * *

## v1.3.0 — 2026-05-21

**Tags:** `omni-cuda-v1.3` · `omni-sagemaker-cuda-v1.3`

**vLLM-Omni source:** [v0.21.0rc1](https://github.com/vllm-project/vllm-omni/releases/tag/v0.21.0rc1) (pre-release, tracking upstream vLLM v0.21.0)

**DLC PR:** [#6110](https://github.com/aws/deep-learning-containers/pull/6110)

### Highlights

- Upgraded to vLLM-Omni 0.21.0rc1, aligned with upstream vLLM v0.21.0
- Cherry-picked upstream Dockerfile fixes for cublas headers (JIT), flashinfer cubin layering, and the `nixl-cu13` install ordering for matching
  `nixl_ep_cpp.so`

### Fixes

- **Voice-clone TTS (Qwen3-TTS-Base) throughput restored** — the upstream Code2Wav decode-chunk un-batching regression flagged in v1.1 is resolved in
  vllm-omni 0.21.0rc1.

### Known Issues

- **Transformers pinned to `<5.9.0`.** Transformers 5.9.0 removed the deprecated `input_embeds` alias and the `cache_position` kwarg from
  `create_causal_mask` / `create_sliding_window_causal_mask`, which breaks Qwen3-TTS decode in vllm-omni 0.21.0rc1. Upstream fix:
  [vllm-project/vllm-omni#3786](https://github.com/vllm-project/vllm-omni/pull/3786). Pin will be dropped once a vllm-omni release containing it
  ships.

* * *

## v1.2.0 — 2026-05-18

**Tags:** `omni-cuda-v1.2` · `omni-sagemaker-cuda-v1.2`

**vLLM-Omni source:** [v0.20.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.20.0) (unchanged from v1.1)

**DLC PR:** [#6101](https://github.com/aws/deep-learning-containers/pull/6101)

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
