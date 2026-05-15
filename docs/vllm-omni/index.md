# vLLM-Omni Inference

Production-ready Docker images for serving multimodal models with [vLLM-Omni](https://github.com/vllm-project/vllm-omni) on {{ aws }}. Built on Amazon Linux 2023 with ongoing security patching.

Supports text-to-speech, audio generation, image generation, video generation, and multimodal chat through OpenAI-compatible APIs.

## Images

| Platform | Image |
|---|---|
| EC2 / EKS | `public.ecr.aws/deep-learning-containers/vllm:omni-cuda` |
| Amazon SageMaker AI | `public.ecr.aws/deep-learning-containers/vllm:omni-sagemaker-cuda` |

For private ECR URIs, see [Image Access](../get_started/index.md).

## Supported Modalities

| Modality | Route | Example Models |
| --- | --- | --- |
| Text-to-Speech | `/v1/audio/speech` | Qwen3-TTS-1.7B, CosyVoice3-0.5B |
| Audio Generation | `/v1/audio/generate` | Stable-Audio-Open-1.0 |
| Image Generation | `/v1/images/generations` | FLUX.2-klein-4B, ERNIE-Image-Turbo |
| Video Generation (async) | `/v1/videos` | Wan2.1-T2V-1.3B, Wan2.1-VACE-1.3B |
| Video Generation (sync) | `/v1/videos/sync` | Wan2.1-T2V-1.3B, Wan2.1-VACE-1.3B |
| Multimodal Chat | `/v1/chat/completions` | Qwen2.5-Omni-3B |

## How We Build

These images are curated builds tracking the [vLLM-Omni](https://github.com/vllm-project/vllm-omni) project:

- **Built from upstream releases** — images track vLLM-Omni releases, each gated by our regression test suite before publication.
- **Regression-tested** — validated against a suite of multimodal models on every release. See [Supported Models](models/index.md) for the full list.
- **Security-patched** — continuously maintained with security patches from {{ aws }} on an Amazon Linux 2023 base.

Each image includes vLLM-Omni (OpenAI-compatible API server with multimodal extensions), PyTorch, CUDA, and NCCL.
