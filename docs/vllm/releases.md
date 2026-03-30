# Releases

This page tracks vLLM {{ dlc_short }} releases, including version history and what changed in each release.

## Latest Release

The latest vLLM {{ dlc_short }} image is available at:

```
public.ecr.aws/deep-learning-containers/vllm:server-cuda
```

## Release History

| Version | Release Date | vLLM Base | PyTorch | CUDA | Highlights |
| --- | --- | --- | --- | --- | --- |
| v1.0.0 | TBD | 0.17.1+ | 2.10.0 | 12.9 | First release with simplified tag format |

> **Note:** This table will be updated as new versions are released. Each entry links to its detailed release notes below.

## Release Notes

### v1.0.0

**Base:** vLLM 0.17.1 + curated patches

**Key packages:**

| Package | Version |
| --- | --- |
| vLLM | 0.17.1+ |
| PyTorch | 2.10.0 |
| CUDA | 12.9 |
| NCCL | 2.27.5 |
| Python | 3.12 |
| EFA | 1.47.0 |

**What's new:**

- Simplified tag format (`server-cuda-v1.0.0`) replacing legacy verbose tags
- 3-part semantic versioning (MAJOR.MINOR.PATCH)
- Curated build with targeted upstream patches for model support and performance
- Available on {{ ecr_public }} and private {{ ecr }}

**What's included from upstream:**

- OpenAI-compatible API server
- Tensor parallelism and pipeline parallelism
- FP8, AWQ, GPTQ quantization support
- LoRA adapter serving
- Structured outputs and tool calling
- Speculative decoding

## Changelog Format

Each release includes:

- **Base** — the upstream vLLM version or commit the build is based on
- **Key packages** — versions of major dependencies (PyTorch, CUDA, NCCL, etc.)
- **What's new** — DLC-specific changes, patches applied, features added
- **What's included from upstream** — notable upstream features available in this build
- **Known issues** — any known limitations (if applicable)

## Notifications

To receive notifications when new vLLM {{ dlc_short }} versions are released, see [Release Notifications](../get_started/release_notifications.md).
