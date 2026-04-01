# Releases

## Latest Release

The latest SGLang AL2023 {{ dlc_short }} image is available at:

```
public.ecr.aws/deep-learning-containers/sglang:server-amzn2023-cuda
```

## Release History

| Version | Release Date | SGLang Base | PyTorch | CUDA | Highlights |
| --- | --- | --- | --- | --- | --- |
| v1.0.0 | TBD | 0.5.9 | 2.9.1 (cu129) | 12.9.1 | First AL2023 from-source build |

## Release Notes

### v1.0.0

**Base:** SGLang 0.5.9 (`v0.5.9` tag), built from source on AL2023

**OS:** Amazon Linux 2023 (`nvidia/cuda:12.9.1-runtime-amzn2023`)

**Key packages:**

| Package | Version |
| --- | --- |
| SGLang | 0.5.9 |
| sgl-kernel | 0.3.21 |
| FlashInfer | 0.6.3 |
| PyTorch | 2.9.1 (cu129) |
| Python | 3.12 |
| CUDA | 12.9.1 |
| cuDNN | 9.16.0.29 |
| NCCL | 2.28.3 |
| EFA | 1.47.0 |
| GDRCopy | 2.5.1 |
| DeepEP | Included (sm90/sm100) |
| Mooncake | 0.3.9 |
| sgl-model-gateway | Rust binary |

**GPU architectures:** Hopper (9.0), Blackwell (10.0, 10.3)

**CVE patches:**

- pillow ≥ 12.1.1
- xgrammar ≥ 0.1.32
- python_multipart ≥ 0.0.22
- setuptools ≥ 78.1.1

**What's new vs Ubuntu variant:**

- Amazon Linux 2023 base image (previously Ubuntu 24.04)
- Full from-source build (SGLang, FlashInfer, DeepEP, Mooncake, sgl-model-gateway)
- sgl-model-gateway compiled as native Rust binary
- Hopper and Blackwell GPU architecture support
- Simplified tag format (`server-amzn2023-cuda`)

**What's included from upstream:**

- OpenAI-compatible API server (`/v1/completions`, `/v1/chat/completions`)
- Tensor parallelism for multi-GPU serving
- FP8 quantization support
- Reasoning parser support for thinking models
- RadixAttention for prefix caching
- Chunked prefill scheduling
- Expert parallelism via DeepEP
- Disaggregated serving via Mooncake

## Changelog Format

Each release includes:

- **Base** — the upstream SGLang version or commit the build is based on
- **Key packages** — versions of major dependencies (PyTorch, CUDA, NCCL, etc.)
- **What's new** — DLC-specific changes, patches applied, features added
- **What's included from upstream** — notable upstream features available in this build
- **Known issues** — any known limitations (if applicable)

## Notifications

To receive notifications when new SGLang {{ dlc_short }} versions are released, see [Release Notifications](../get_started/release_notifications.md).

## See Also

- [Versioning](versioning.md) — tag format and semantic versioning
- [Support Policy](support_policy.md) — lifecycle and patch delivery
