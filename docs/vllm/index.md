# vLLM on {{ dlc_long }}

{{ dlc_long }} provide pre-built, optimized Docker images for running [vLLM](https://docs.vllm.ai/) inference workloads on {{ aws }} infrastructure.

## How We Build

The vLLM {{ dlc_short }} images are **curated builds**, not simple repackages of upstream releases:

- **Built from a chosen base reference** — a specific commit, release candidate, or point in the vLLM repository history — with targeted patches
  applied from upstream PRs, forks, and community contributions for new model support, bug fixes, and performance improvements.
- **Opinionated testing** — validated against a selected suite of model-serving use cases relevant to {{ aws }} customers, rather than relying solely
  on upstream's broader test suite.
- **Faster access with higher confidence** — delivers the latest advancements while maintaining reliability for real-world workloads. When regressions
  are caught, we troubleshoot and contribute fixes upstream or apply local patches rather than waiting for upstream releases.

## What's Included

Each vLLM {{ dlc_short }} image ships with:

- **vLLM** inference engine with OpenAI-compatible API server
- **PyTorch** and **CUDA** optimized for NVIDIA GPUs
- **NCCL** for multi-GPU communication
- **EFA** support for multi-node deployments on {{ ec2 }}
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
- [Release Notes](../releasenotes/vllm-server/index.md) — Version history and release notes
- [Available Images](../reference/available_images.md) — All published image URIs
- [Support Policy](../reference/support_policy.md) — GA/EOP lifecycle and patching

## Upstream Documentation

For general vLLM usage beyond the {{ dlc_short }} images, see the [vLLM documentation](https://docs.vllm.ai/).
