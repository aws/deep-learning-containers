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
| v1.0.0 | TBD | 0.17.1+ | 2.6.0 | 12.9.1 | First release with simplified tag format, AL2023 base |

> **Note:** This table will be updated as new versions are released. Each entry links to its detailed release notes below.

## Release Notes

### v1.0.0

**Base:** vLLM 0.17.1 (`v0.17.1` tag) + curated patches, built as `0.17.1+amzn2023`

**OS:** Amazon Linux 2023 (`nvidia/cuda:12.9.1-runtime-amzn2023`)

**Key packages:**

| Package | Version |
| --- | --- |
| vLLM | 0.17.1+amzn2023 |
| Python | 3.12 |
| CUDA | 12.9.1 |
| FlashInfer | 0.6.4 |
| EFA | 1.47.0 |

**GPU architectures:** V100 (7.0), T4 (7.5), A100 (8.0), L4/L40S (8.9), H100 (9.0), B200 (10.0), B300 (12.0)

**Extras included:**

- DeepGEMM (for sm90a/sm100a)
- DeepEP (for sm90a/sm100a)
- accelerate, hf_transfer, bitsandbytes, modelscope, runai-model-streamer

**CVE patches:**

- pillow ≥ 12.1.1
- xgrammar ≥ 0.1.32
- PyJWT ≥ 2.12.0
- cbor2 ≥ 5.9.0

**What's new:**

- Amazon Linux 2023 base image (previously Ubuntu)
- Simplified tag format (`server-cuda-v1.0.0`) replacing legacy verbose tags
- 3-part semantic versioning (MAJOR.MINOR.PATCH)
- Curated build from `v0.17.1` with targeted upstream patches
- DeepGEMM and DeepEP for H100/B200 performance
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
