# Versioning

Understand the vLLM {{ dlc_short }} image tag format and how to select the right image.

## Image Tag Format

vLLM {{ dlc_short }} image tags follow this pattern:

```
<vllm_version>-gpu-<python>-<cuda>-<os>-<platform>
```

| Component | Description | Example |
| --- | --- | --- |
| `vllm_version` | Upstream vLLM release version | `0.17.1` |
| `gpu` | Accelerator type (always `gpu` for vLLM) | `gpu` |
| `python` | Python version | `py312` |
| `cuda` | CUDA toolkit version | `cu129` |
| `os` | Operating system | `ubuntu22.04` |
| `platform` | Target {{ aws }} platform | `ec2` or `sagemaker` |

### Example Tags

| Tag | Platform |
| --- | --- |
| `0.17.1-gpu-py312-cu129-ubuntu22.04-ec2` | {{ ec2_short }}, {{ ecs_short }}, {{ eks_short }} |
| `0.17.1-gpu-py312-cu129-ubuntu22.04-sagemaker` | {{ sm_short }} |

## Tag Aliases

Each image has multiple tags for convenience:

| Tag Type | Example | Description |
| --- | --- | --- |
| Full | `0.17.1-gpu-py312-cu129-ubuntu22.04-ec2` | Pinned to exact version |
| Short | `0.17.1-gpu-py312-ec2` | Omits CUDA and OS |
| Minor | `0.17-gpu-py312-cu129-ubuntu22.04-ec2-v1` | Tracks latest patch in minor series |
| Minor short | `0.17-gpu-py312-ec2` | Shortest minor alias |

!!! tip "Recommendation" Use the **full tag** in production for reproducibility. Use **minor aliases** in development to automatically pick up patch
releases.

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
763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm:0.17.1-gpu-py312-cu129-ubuntu22.04-ec2

# EU (Ireland)
763104351884.dkr.ecr.eu-west-1.amazonaws.com/vllm:0.17.1-gpu-py312-cu129-ubuntu22.04-ec2
```
