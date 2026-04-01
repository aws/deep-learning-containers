# Versioning

## Build Philosophy

The SGLang AL2023 {{ dlc_short }} images are **curated from-source builds** — not direct repackages of upstream SGLang releases. Each image:

- Starts from a chosen tag in the SGLang repository
- Compiles all components from source on Amazon Linux 2023
- Is validated against a selected suite of model-serving use cases relevant to {{ aws }} customers

## Simplified Tag Format

SGLang AL2023 {{ dlc_short }} images use a simplified tagging format:

```
server-amzn2023-cuda
```

With versioned variants for each platform:

**{{ ec2_short }} / {{ ecs_short }} / {{ eks_short }}:**

| Tag | Example | Description |
| --- | --- | --- |
| Base | `server-amzn2023-cuda` | Latest release (rolling) |
| Full version | `server-amzn2023-cuda-v1.0.0` | Pinned to exact release |
| Minor | `server-amzn2023-cuda-v1.0` | Latest patch in v1.0.x series |
| Major | `server-amzn2023-cuda-v1` | Latest release in v1.x.x series |

**{{ sm_short }}:**

| Tag | Example | Description |
| --- | --- | --- |
| Base | `server-amzn2023-sagemaker-cuda` | Latest release (rolling) |
| Full version | `server-amzn2023-sagemaker-cuda-v1.0.0` | Pinned to exact release |
| Minor | `server-amzn2023-sagemaker-cuda-v1.0` | Latest patch in v1.0.x series |
| Major | `server-amzn2023-sagemaker-cuda-v1` | Latest release in v1.x.x series |

## Semantic Versioning

The version follows 3-part semantic versioning (`MAJOR.MINOR.PATCH`):

| Increment | When |
| --- | --- |
| **MAJOR** | CUDA/Python breaking changes, removed model support, breaking server features |
| **MINOR** | CUDA/Python updates, new features, new model support, bug fixes |
| **PATCH** | Security patches, backwards-compatible bug fixes |

> **Tip:** Use the **full version tag** (`server-amzn2023-cuda-v1.0.0`) in production for reproducibility. Use the **minor tag**
> (`server-amzn2023-cuda-v1.0`) in development to automatically pick up security patches.

## Image URI Format

### {{ ecr_public }} (Recommended)

```
public.ecr.aws/deep-learning-containers/sglang:<tag>
```

### Private {{ ecr }}

```
<account_id>.dkr.ecr.<region>.amazonaws.com/sglang:<tag>
```

## Legacy Tags

Older SGLang {{ dlc_short }} images used a verbose tag format that encoded Python, CUDA, and OS versions:

```
<version>-gpu-py312-cu129-<os>-<platform>
```

These legacy images remain available for backward compatibility. The new `server-amzn2023-cuda` format coexists alongside them in the same `sglang`
ECR repository.

## See Also

- [Releases](releases.md) — release history and notes
- [Support Policy](support_policy.md) — lifecycle and patch delivery
