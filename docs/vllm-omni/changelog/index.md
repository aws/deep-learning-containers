# Changelog

Changelog for the Amazon Linux 2023-based vLLM-Omni images (`omni-cuda`, `omni-sagemaker-cuda`).

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
