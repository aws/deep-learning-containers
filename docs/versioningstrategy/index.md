# Versioning Strategy

## Image Naming Conventions

AWS Deep Learning Containers (DLC) images fall into two naming categories based on their release model:

**Product-versioned images** (Ray, model-based DLCs, etc.) These images use a product-level name without embedding the framework version. Framework
upgrades are reflected as major version bumps rather than new image streams.

| Image name | Tag example |
| --- | --- |
| `ray` | `serve-ml-cuda-v1.0.0` |

**Framework-versioned images** (classic DLCs: vLLM, PyTorch Training/Inference, etc.) These images embed the framework version, CUDA version, and OS
in the tag. A date string can be appended to pin a specific build.

**Note:** The tag also encodes compute type and deployment target. A `cu<version>` component (e.g., `cu130`) indicates a GPU image; `cpu` indicates a
CPU-only image — for example, `2.11-cu130-amzn2023` and `2.11-cpu-amzn2023`. No platform suffix means the image targets EC2; a `sagemaker` suffix
indicates the image is built for SageMaker.

| Image name | Tag example |
| --- | --- |
| `pytorch` | `2.11-cu130-amzn2023` |
| `pytorch` | `2.11-cpu-amzn2023` |

Product-versioned images follow semantic versioning. Framework-versioned images use the framework version, CUDA version, and OS as the tag, with an
optional date string for pinning specific builds.

## Version Bumps

The following version bump rules apply to product-versioned images. Framework-versioned images do not use DLC version numbers — see
[Tag Aliases](#tag-aliases) below.

- **MAJOR** — [Backwards-incompatible](#backwards-compatibility) changes. This includes core component major or minor version bumps (e.g., CUDA,
  Python, framework), API removals, and changes to default behavior.
- **MINOR** — [Backwards-compatible](#backwards-compatibility) feature updates and improvements. This includes core component patch version bumps
  (e.g., CUDA, Python, framework), supporting dependency updates, and new functionality that does not alter existing behavior. (See
  [Backwards Compatibility](#backwards-compatibility) definition below)
- **PATCH** — Security patches and [backwards-compatible](#backwards-compatibility) bug fixes. No new features.

**Note:** Existing DLC images use a legacy tag format that encodes processor, Python version, CUDA version, and OS in the tag (e.g.,
`vllm:0.17.0-gpu-py312-cu129-ubuntu22.04-ec2-v1.2`). New framework-versioned images use a simplified tag with framework version, CUDA version, and OS
(e.g., `vllm:0.19.1-cu130-amzn2023`). Product-versioned images use semantic versioning (`v<MAJOR>.<MINOR>.<PATCH>`). For framework-versioned images,
changes to framework version, CUDA, or OS create a new tag. For product-versioned images, all breaking changes result in a major version bump.

| Version Change | Framework-versioned | Product-versioned |
| --- | --- | --- |
| PyTorch 2.9 → 2.10 | New tag (e.g., `vllm:0.19.1-cu130-amzn2023`) | Major bump, `v1.x.x` → `v2.0.0` |

## Backwards Compatibility

A change is considered backwards-compatible if it passes the full DLC test suite without any changes. That suite consists of three layers:

- **Unit tests** — Verify that individual modules and functions behave as documented. Covers API signatures, default parameter values, and return
  types.
- **Functional tests** — Validate end-to-end workflows within a single container, such as model training, inference, etc. Confirms that expected
  inputs continue to produce expected outputs.
- **Integration tests** — Run multi-component scenarios that reflect real-world usage, including distributed training, serving endpoints, and
  interaction with AWS services. Validates that the container works correctly in its broader deployment context.

A change is classified as backwards-incompatible — and therefore requires a major version bump — if it causes any existing unit, functional, or
integration test to fail.

## Tag Aliases

Each image is published with multiple tags at different levels of specificity. Choose the tag that matches your stability requirements.

### Product-versioned images

| Tag | Example | Tracks | Updates when |
| --- | --- | --- | --- |
| `<image>` | `base` | Latest across all versions | Any release, including major versions with breaking changes |
| `<image>-v<MAJOR>` | `base-v1` | Latest within a major version | Minor or patch release — [backwards-compatible](#backwards-compatibility) feature updates, security patches, and bug fixes |
| `<image>-v<MAJOR>.<MINOR>` | `base-v1.0` | Latest within a minor version | Patch release — [backwards-compatible](#backwards-compatibility) security patches and bug fixes; no new features |
| `<image>-v<MAJOR>.<MINOR>.<PATCH>` | `base-v1.0.0` | Pinned — never changes | Never — immutable snapshot of a specific release |

### Framework-versioned images

| Tag | Example | Tracks | Updates when |
| --- | --- | --- | --- |
| `<image>:<fwk_version>-<cuda_version>-<os>` | `vllm:0.19.1-cu130-amzn2023` | Latest build for this framework/CUDA/OS combo | Any new build |
| `<image>:<fwk_version>-<cuda_version>-<os>-<date>` | `vllm:0.19.1-cu130-amzn2023-20260430` | Pinned — never changes | Never — immutable snapshot |

## Choosing a Tag

### Product-versioned images

- **`<image>`** — Always get the latest image, including new major versions. Use this when you want the newest features and dependencies and can adapt
  to breaking changes. Not recommended for production environments, as pulling this tag may introduce breaking changes without warning.
- **`<image>-v<MAJOR>`** — Stay current within a major version. Includes [backwards-compatible](#backwards-compatibility) feature updates, security
  patches, and bug fixes.
- **`<image>-v<MAJOR>.<MINOR>`** — Receive [backwards-compatible](#backwards-compatibility) security patches and bug fixes. No new features. Use this
  when stability matters more than new features.
- **`<image>-v<MAJOR>.<MINOR>.<PATCH>`** — Pinned to an exact release. The image never changes. Use this when you need a fully reproducible
  environment or want to control exactly when upgrades happen.

### Framework-versioned images

- **`<image>:<fwk_version>-<cuda_version>-<os>`** — Latest build for this framework/CUDA/OS combination. Updated with each new build. Use when you
  want the latest patches and fixes for a specific framework version.
- **`<image>:<fwk_version>-<cuda_version>-<os>-<date>`** — Pinned to a specific build date. The image never changes. Use for reproducible
  environments.
