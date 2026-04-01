# Support Policy

## Lifecycle Definitions

| Term | Definition |
| --- | --- |
| **GA (General Availability)** | The date when an SGLang {{ dlc_short }} version becomes officially supported for production use. |
| **EOP (End of Patch)** | The date after which an SGLang {{ dlc_short }} version no longer receives security patches or bug fixes. |

> **Note:** The SGLang AL2023 {{ dlc_short }} is currently in **gamma**. GA release is pending.

## Support Timeline

Each SGLang {{ dlc_short }} version is supported from its GA date until its EOP date. During this window, {{ aws }} provides:

- Security patches for critical and high-severity vulnerabilities
- Bug fixes for container-level issues
- Compatibility updates for {{ aws }} services

## How Patches Are Delivered

Because SGLang {{ dlc_short }} images are curated from-source builds, patching follows a different model than upstream:

- When regressions or vulnerabilities are identified, we troubleshoot and contribute fixes upstream or apply local patches
- Patches are delivered as new image builds — we do not wait for upstream releases
- Security patches increment the **PATCH** version (e.g., `v1.0.0` → `v1.0.1`)
- The minor tag (e.g., `server-amzn2023-cuda-v1.0`) automatically points to the latest patch release

## Current Support Status

For the current GA and EOP dates of all SGLang {{ dlc_short }} versions, see the [Framework Support Policy](../reference/support_policy.md) page.

For all available image tags and versions, see [Available Images](../reference/available_images.md).

## Upgrade Guidance

We recommend upgrading to the latest supported SGLang {{ dlc_short }} version to benefit from:

- Performance improvements and new model support from upstream and curated patches
- Latest security patches
- Updated CUDA, PyTorch, and NCCL versions

### Upgrade Steps

1. Review the [Release Notes](releases.md) for changes between versions
2. Update your image tag to the new version
3. Test with your model and workload before deploying to production

> **Warning:** SGLang does not guarantee API stability across minor versions. Review the
> [SGLang changelog](https://github.com/sgl-project/sglang/releases) for breaking changes before upgrading.

## End of Patch Availability

After an SGLang {{ dlc_short }} version reaches its EOP date:

- Container images remain available on {{ ecr }} and the {{ ecr_public }}
- No further security patches or bug fixes will be provided
- We recommend migrating to a supported version

## See Also

- [Releases](releases.md) — release history and notes
- [Versioning](versioning.md) — tag format and semantic versioning
