# vLLM on {{ dlc_long }}

{{ dlc_long }} provide pre-built, optimized Docker images for running [vLLM](https://docs.vllm.ai/) inference workloads on {{ aws }} infrastructure.
These images are tested, patched, and maintained by {{ aws }}, so you can focus on serving models rather than managing dependencies.

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
- [Features](features.md) — Quantization, LoRA, structured outputs, and more
- [Versioning](versioning.md) — Image tag format and how to pick the right image
- [Support Policy](support_policy.md) — GA/EOP lifecycle and patching

## Upstream Documentation

For general vLLM usage beyond the {{ dlc_short }} images, see the [vLLM documentation](https://docs.vllm.ai/).
