# Versioning

Understand the vLLM {{ dlc_short }} image versioning and simplified tag format.

## Build Philosophy

The vLLM {{ dlc_short }} images are **curated builds** — not direct repackages of upstream vLLM releases. Each image:

- Starts from a chosen base reference in the vLLM repository (a specific commit, release candidate, or branch point)
- Applies targeted patches from upstream PRs, forks, and community contributions
- Is validated against a selected suite of model-serving use cases relevant to {{ aws }} customers

The version number (e.g., `0.17.1`) indicates the upstream vLLM version the build is based on, but the image may include additional fixes and features
not yet in that upstream release.

## Simplified Tag Format

vLLM {{ dlc_short }} images use a simplified tagging format. Details like Python version, CUDA version, and OS type are documented in release
materials (release notes, changelogs, available images tables) rather than encoded in the tag.

```
<vllm_version>-gpu-<platform>
```

| Component | Description | Example |
| --- | --- | --- |
| `vllm_version` | Base upstream vLLM version | `0.17.1` |
| `gpu` | Accelerator type | `gpu` |
| `platform` | Target {{ aws }} platform | `ec2` or `sagemaker` |

### Example Tags

| Tag | Platform |
| --- | --- |
| `0.17.1-gpu-ec2` | {{ ec2_short }}, {{ ecs_short }}, {{ eks_short }} |
| `0.17.1-gpu-sagemaker` | {{ sm_short }} |

!!! info Detailed package versions (Python, CUDA, cuDNN, NCCL, etc.) are listed in the [Release Notes](../releasenotes/vllm/index.md) and
[Available Images](../reference/available_images.md) tables for each release.

## Platform Selection

| Platform Tag | Use With |
| --- | --- |
| `ec2` | {{ ec2 }}, {{ ecs }}, {{ eks }}, or any Docker environment |
| `sagemaker` | {{ sagemaker }} inference endpoints only |

The `sagemaker` images include the {{ sm_short }} inference toolkit and entrypoint. The `ec2` images run the vLLM server directly.

## Version History

| vLLM Version | PyTorch | CUDA | Key Changes |
| --- | --- | --- | --- |
| 0.17.1 | 2.10.0 | 12.9 | Latest release |
| 0.17.0 | 2.10.0 | 12.9 | PyTorch 2.10 upgrade |
| 0.16.0 | 2.9.1 | 12.9 | EFA 1.47.0 |
| 0.15.1 | 2.9.1 | 12.9 | — |
| 0.14.0 | 2.9.1 | 12.9.1 | FlashInfer 0.5.3 |
| 0.13.0 | 2.9.0 | 12.9.1 | Initial Python 3.12 support |

For the full list of available images and tags, see [Available Images](../reference/available_images.md).

## Forming the Image URI

```
<account_id>.dkr.ecr.<region>.amazonaws.com/vllm:<tag>
```

See [Region Availability](../reference/available_images.md#region-availability) for account IDs per region.

### Example

```bash
# US West (Oregon)
763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm:0.17.1-gpu-ec2

# EU (Ireland)
763104351884.dkr.ecr.eu-west-1.amazonaws.com/vllm:0.17.1-gpu-ec2
```
