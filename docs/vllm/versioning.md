# Versioning

Understand the vLLM {{ dlc_short }} image versioning and simplified tag format.

## Build Philosophy

The vLLM {{ dlc_short }} images are **curated builds** — not direct repackages of upstream vLLM releases. Each image:

- Starts from a chosen base reference in the vLLM repository (a specific commit, release candidate, or branch point)
- Applies targeted patches from upstream PRs, forks, and community contributions
- Is validated against a selected suite of model-serving use cases relevant to {{ aws }} customers

## Simplified Tag Format

vLLM {{ dlc_short }} images use a simplified tagging format. Details like Python version, CUDA version, and OS type are documented in release
materials (release notes, changelogs, available images tables) rather than encoded in the tag.

The tag format is:

```
server-cuda
```

With versioned variants:

| Tag | Example | Description |
| --- | --- | --- |
| Base | `server-cuda` | Latest release (rolling) |
| Full version | `server-cuda-v1.0.0` | Pinned to exact release |
| Minor | `server-cuda-v1.0` | Latest patch in v1.0.x series |
| Major | `server-cuda-v1` | Latest release in v1.x.x series |

When a platform-specific variant exists (e.g., for Bedrock), the platform is inserted between `server` and `cuda`:

| Tag | Example |
| --- | --- |
| Base | `server-bedrock-cuda` |
| Full version | `server-bedrock-cuda-v1.0.0` |

!!! info Detailed package versions (Python, CUDA, cuDNN, NCCL, etc.) are listed in the [Release Notes](../releasenotes/vllm/index.md) and
[Available Images](../reference/available_images.md) tables for each release.

## Semantic Versioning

The version follows 3-part semantic versioning (`MAJOR.MINOR.PATCH`):

| Increment | When |
| --- | --- |
| **MAJOR** | CUDA/Python breaking changes, removed model support, breaking server features |
| **MINOR** | CUDA/Python updates, new features, new model support, bug fixes |
| **PATCH** | Security patches, backwards-compatible bug fixes |

!!! tip "Recommendation" Use the **full version tag** (`server-cuda-v1.0.0`) in production for reproducibility. Use the **minor tag**
(`server-cuda-v1.0`) in development to automatically pick up security patches.

## Pulling Images

vLLM {{ dlc_short }} images are available from both the {{ ecr_public }} and private {{ ecr }} registries.

### {{ ecr_public }} (Recommended)

The simplest way to pull images — no authentication required:

```bash
docker pull public.ecr.aws/deep-learning-containers/vllm:server-cuda-v1.0.0
```

### Private {{ ecr }}

For use within {{ aws }} accounts. Requires authentication and uses a region-specific URI:

```
<account_id>.dkr.ecr.<region>.amazonaws.com/vllm:<tag>
```

See [Region Availability](../reference/available_images.md#region-availability) for account IDs per region.

```bash
# Authenticate
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com

# Pull
docker pull 763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm:server-cuda-v1.0.0
```

### Examples

=== "{{ ecr_public }}"

````
```bash
# Pinned to exact version
docker pull public.ecr.aws/deep-learning-containers/vllm:server-cuda-v1.0.0

# Latest patch in v1.0 series
docker pull public.ecr.aws/deep-learning-containers/vllm:server-cuda-v1.0

# Latest v1 release
docker pull public.ecr.aws/deep-learning-containers/vllm:server-cuda-v1
```
````

=== "Private {{ ecr }}"

````
```bash
# US West (Oregon)
docker pull 763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm:server-cuda-v1.0.0

# EU (Ireland)
docker pull 763104351884.dkr.ecr.eu-west-1.amazonaws.com/vllm:server-cuda-v1.0.0
```
````

## Legacy Tags

Older vLLM {{ dlc_short }} images used a verbose tag format that encoded Python, CUDA, and OS versions:

```
0.17.1-gpu-py312-cu129-ubuntu22.04-ec2
```

These legacy images remain available. The new `server-cuda` format coexists alongside them in the same `vllm` ECR repository.
