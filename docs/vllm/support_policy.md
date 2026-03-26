# Support Policy

This page details the support lifecycle for vLLM {{ dlc_short }} images.

## Lifecycle Definitions

| Term | Definition |
| --- | --- |
| **GA (General Availability)** | The date when a vLLM {{ dlc_short }} version becomes officially supported and available for production use. |
| **EOP (End of Patch)** | The date after which a vLLM {{ dlc_short }} version no longer receives security patches or bug fixes. |

## Support Timeline

Each vLLM {{ dlc_short }} version is supported from its GA date until its EOP date. During this window, {{ aws }} provides:

- Security patches for critical and high-severity vulnerabilities
- Bug fixes for container-level issues
- Compatibility updates for {{ aws }} services

## Current Support Status

For the current GA and EOP dates of all vLLM {{ dlc_short }} versions, see the [Framework Support Policy](../reference/support_policy.md) page.

For all available image tags and versions, see [Available Images](../reference/available_images.md).

## Upgrade Guidance

We recommend upgrading to the latest supported vLLM {{ dlc_short }} version to benefit from:

- Performance improvements in newer vLLM releases
- Expanded model support
- Latest security patches
- Updated CUDA, PyTorch, and NCCL versions

### Upgrade Steps

1. Review the [Version History](versioning.md#version-history) for changes between versions
2. Update your image tag to the new version
3. Test with your model and workload before deploying to production

!!! warning vLLM does not guarantee API stability across minor versions. Review the [vLLM changelog](https://github.com/vllm-project/vllm/releases)
for breaking changes before upgrading.

## End of Patch Availability

After a vLLM {{ dlc_short }} version reaches its EOP date:

- Container images remain available on {{ ecr }} and the {{ ecr_public }}
- No further security patches or bug fixes will be provided
- We recommend migrating to a supported version

## Patching Policy

Within a supported version's lifecycle:

- **Security patches** are applied as new image builds with the same version tag
- **Minor alias tags** (e.g., `0.17-gpu-py312-ec2`) are updated to point to the latest patched build
- **Full version tags** (e.g., `0.17.1-gpu-py312-cu129-ubuntu22.04-ec2`) remain immutable

!!! tip Use minor alias tags in non-production environments to automatically receive patches. Pin full version tags in production for stability.
