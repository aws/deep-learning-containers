# Multimodal Serving using vLLM-Omni DLC

Production-ready Docker images for serving multimodal models with [vLLM-Omni](https://github.com/vllm-project/vllm-omni) on {{ aws }}. Built on Amazon
Linux 2023 with ongoing security patching.

Supports text-to-speech, audio generation, image generation, video generation, and multimodal chat through OpenAI-compatible APIs.

## Images

| Platform | Image | Default Port |
| --- | --- | --- |
| EC2 / EKS | `public.ecr.aws/deep-learning-containers/vllm:omni-cuda` | 8080 |
| Amazon SageMaker AI | `public.ecr.aws/deep-learning-containers/vllm:omni-sagemaker-cuda` | 8080 |

All images are also available on the [ECR Public Gallery](https://gallery.ecr.aws/deep-learning-containers/vllm). For private ECR URIs, see
[Image Access](../get_started/index.md).

## Supported Modalities

| Modality | Route | Example Models |
| --- | --- | --- |
| Text-to-Speech | `/v1/audio/speech` | Qwen3-TTS-1.7B, CosyVoice3-0.5B |
| Audio Generation | `/v1/audio/generate` | Stable-Audio-Open-1.0 |
| Image Generation | `/v1/images/generations` | FLUX.2-klein-4B, ERNIE-Image-Turbo |
| Video Generation (async) | `/v1/videos` | Wan2.1-T2V-1.3B, Wan2.1-VACE-1.3B |
| Video Generation (sync) | `/v1/videos/sync` | Wan2.1-T2V-1.3B, Wan2.1-VACE-1.3B |
| Multimodal Chat | `/v1/chat/completions` | Qwen2.5-Omni-3B |

## What's Included

In addition to vLLM-Omni and its core stack (PyTorch, CUDA 13.0, NCCL, Python 3.12), the images bundle:

- **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)** — fused attention kernels with precompiled cubins for fast cold start
- **[DeepEP](https://github.com/deepseek-ai/DeepEP)** — expert-parallel kernels for large MoE models
- **[LMCache](https://github.com/LMCache/LMCache) + [NIXL](https://github.com/ai-dynamo/nixl)** — KV-cache offloading and disaggregated prefill/decode
- **[runai-model-streamer](https://github.com/run-ai/runai-model-streamer)** — stream model weights directly from S3 or GCS
- **[EFA](https://aws.amazon.com/hpc/efa/) and [OpenMPI](https://www.open-mpi.org/)** — high-throughput multi-node networking on supported instances
- **espeak-ng and ffmpeg** — system-level dependencies for TTS phonemizer and audio/video encoding

The SageMaker image additionally includes a routing middleware that dispatches `/invocations` to omni-specific routes (TTS, image, video, etc.) via
the `CustomAttributes` header. See [Amazon SageMaker AI Deployment](deployment/sagemaker.md).

## CUDA Forward Compatibility

The images use CUDA 13.0, which requires NVIDIA driver 580+ on the host. On hosts with older datacenter drivers, set
`-e VLLM_ENABLE_CUDA_COMPATIBILITY=1` to enable the bundled CUDA 13 forward-compat layer.

## How We Build

These images are curated builds tracking the [vLLM-Omni](https://github.com/vllm-project/vllm-omni) project:

- **Built from upstream releases** — images track vLLM-Omni releases, each gated by our regression test suite before publication.
- **Regression-tested** — validated against a suite of multimodal models on every release. See [Supported Models](models/index.md) for the full list.
- **Security-patched** — continuously maintained with security patches from {{ aws }} on an Amazon Linux 2023 base.
