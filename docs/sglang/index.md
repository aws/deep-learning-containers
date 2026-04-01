# SGLang on {{ dlc_long }}

{{ dlc_long }} provide pre-built, optimized Docker images for running [SGLang](https://docs.sglang.ai/) inference workloads on {{ aws }}
infrastructure.

## How We Build

The SGLang AL2023 {{ dlc_short }} images are **curated from-source builds**, not repackages of upstream releases:

- **Built entirely from source** on Amazon Linux 2023 — SGLang, sgl-kernel, FlashInfer, DeepEP, Mooncake, and sgl-model-gateway (Rust binary) are
  compiled from source against CUDA 12.9.1.
- **Opinionated testing** — validated against a selected suite of model-serving use cases relevant to {{ aws }} customers, including smoke tests,
  benchmarks, and SageMaker endpoint tests.
- **Faster access with higher confidence** — delivers the latest advancements while maintaining reliability. When regressions are caught, we
  troubleshoot and contribute fixes upstream or apply local patches.

## What's Included

Each SGLang AL2023 {{ dlc_short }} image ships with:

- **SGLang** inference engine with OpenAI-compatible API server
- **sgl-model-gateway** — Rust binary for model routing
- **PyTorch** with CUDA 12.9.1 support
- **FlashInfer** and **sgl-kernel** for optimized attention
- **DeepEP** for expert parallelism on Hopper/Blackwell GPUs
- **Mooncake** transfer engine for disaggregated serving
- **NCCL** and **EFA** for multi-GPU/multi-node communication
- Security patches and dependency updates from {{ aws }}

## Available Platforms

| Platform | Use Case |
| --- | --- |
| {{ ec2_short }}, {{ ecs_short }}, {{ eks_short }} | Self-managed inference on GPU instances |
| {{ sm_short }} | Managed inference endpoints with auto-scaling |

## Quick Links

- [Quickstart](quickstart.md) — Pull an image and run your first inference
- [Configuration](configuration.md) — Environment variables and server arguments
- [Deployment](deployment.md) — Deploy on {{ ec2 }}, {{ ecs }}, {{ eks }}, or {{ sagemaker }}
- [Supported Models](supported_models.md) — Models tested with the {{ dlc_short }} images
- [Benchmarks](benchmarks.md) — Performance numbers on {{ aws }} GPU instances
- [Releases](releases.md) — Release history and release notes
- [Versioning](versioning.md) — Simplified tag format and how to pick the right image
- [Support Policy](support_policy.md) — GA/EOP lifecycle and patching

## Upstream Documentation

For general SGLang usage beyond the {{ dlc_short }} images, see the [SGLang documentation](https://docs.sglang.ai/).
