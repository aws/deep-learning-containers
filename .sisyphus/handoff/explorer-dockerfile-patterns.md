# DLC Dockerfile Patterns — vLLM & SGLang

## 1. Directory Structure

```
docker/
├── vllm/
│   ├── Dockerfile           (4.7KB — Ubuntu-based, FROM upstream image)
│   └── Dockerfile.amzn2023  (12KB — AL2023, multi-stage build from source)
├── sglang/
│   ├── Dockerfile           (5.5KB — Ubuntu-based, FROM upstream image)
│   └── Dockerfile.amzn2023  (15KB — AL2023, multi-stage build from source)
├── pytorch/
├── ray/
└── lambda/
    └── .dockerignore        (only .dockerignore in repo)

scripts/
├── common/
│   ├── install_efa.sh            (Ubuntu EFA installer)
│   ├── install_efa_amzn2023.sh   (AL2023 EFA installer)
│   ├── setup_oss_compliance.sh   (OSS compliance — downloads zip, generates compliance)
│   └── start_cuda_compat.sh      (CUDA forward-compat loader)
├── telemetry/
│   ├── deep_learning_container.py    (Python telemetry — sends usage to AWS)
│   └── bash_telemetry.sh.template    (Shell wrapper with {{FRAMEWORK}}/{{FRAMEWORK_VERSION}}/{{CONTAINER_TYPE}} placeholders)
├── vllm/
│   ├── dockerd_entrypoint.sh         (EC2: runs `python3 -m vllm.entrypoints.openai.api_server "$@"`)
│   ├── sagemaker_entrypoint.sh       (SM: parses SM_VLLM_* env vars → CLI args, auto-detects model)
│   └── amzn2023/patches/.gitkeep     (patch directory for AL2023 source builds)
└── sglang/
    ├── dockerd_entrypoint.sh         (EC2: runs `python3 -m sglang.launch_server "$@"`)
    └── sagemaker_entrypoint.sh       (SM: parses SM_SGLANG_* env vars → CLI args, defaults port 8080)

.github/config/
├── vllm-ec2.yml              (build config for vLLM EC2)
├── vllm-sagemaker.yml
├── vllm-ec2-amzn2023.yml
├── sglang-ec2.yml
├── sglang-sagemaker.yml
└── sglang-ec2-amzn2023.yml
```

## 2. Two Dockerfile Variants

### Variant A: Ubuntu-based (`Dockerfile`) — Overlay on upstream image

Both vLLM and SGLang use the same pattern:

```dockerfile
ARG BASE_IMAGE=<upstream-image>:<version>
FROM $BASE_IMAGE AS base
# ... DLC overlay (deps, telemetry, EFA, OSS compliance) ...
FROM base AS <framework>-ec2        # EC2 target
FROM base AS <framework>-sagemaker  # SageMaker target
```

- **vLLM base**: `vllm/vllm-openai:v0.17.1` (Docker Hub)
- **SGLang base**: `lmsysorg/sglang:v0.5.9` (Docker Hub)
- Single-stage overlay: installs DLC deps on top of upstream pre-built image
- Uses `apt-get` (Ubuntu)

### Variant B: AL2023-based (`Dockerfile.amzn2023`) — Multi-stage from source

Both use multi-stage builds on `nvidia/cuda:*-amzn2023`:

```
vLLM AL2023:  source → build → deps → runtime → base(DLC overlay) → ec2/sagemaker targets
SGLang AL2023: builder → runtime → ec2/sagemaker targets
```

- Clones upstream repo, compiles from source
- Uses `dnf` (AL2023)
- Separate build and runtime stages for smaller final image
- vLLM applies patches from `scripts/vllm/amzn2023/patches/`

## 3. Common DLC Overlay Pattern (shared across ALL Dockerfiles)

Every Dockerfile applies the same DLC customization layer in this order:

```dockerfile
# 1. Labels
LABEL maintainer="Amazon AI"
LABEL dlc_major_version="1"

# 2. Standard ENV block
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DLC_CONTAINER_TYPE=general \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib:/opt/amazon/openmpi/lib:/opt/amazon/efa/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    PATH="/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:/usr/local/cuda/bin:${PATH}"

# 3. Install botocore (DLC dependency)
RUN pip install botocore

# 4. CVE patches (version-pinned pip packages)
RUN pip install "pillow>=12.1.1" "xgrammar>=0.1.32" ...

# 5. Copy telemetry scripts + template substitution
COPY ./scripts/telemetry/deep_learning_container.py /usr/local/bin/
COPY ./scripts/telemetry/bash_telemetry.sh.template /tmp/
ARG FRAMEWORK
ARG FRAMEWORK_VERSION
ARG CONTAINER_TYPE
RUN sed -e "s/{{FRAMEWORK}}/${FRAMEWORK}/g" \
        -e "s/{{FRAMEWORK_VERSION}}/${FRAMEWORK_VERSION}/g" \
        -e "s/{{CONTAINER_TYPE}}/${CONTAINER_TYPE}/g" \
        /tmp/bash_telemetry.sh.template > /usr/local/bin/bash_telemetry.sh

# 6. OSS compliance
COPY ./scripts/common/setup_oss_compliance.sh setup_oss_compliance.sh
RUN bash setup_oss_compliance.sh ${PYTHON} && rm setup_oss_compliance.sh

# 7. EFA installation
COPY ./scripts/common/install_efa.sh install_efa.sh
RUN bash install_efa.sh ${EFA_VERSION} && rm install_efa.sh

# 8. nvjpeg CVE fix (replace vulnerable libnvjpeg with patched version)

# 9. Cleanup
RUN rm -rf /root/.cache /tmp/* /var/lib/apt/lists/*
```

## 4. Build Args Pattern

### Required ARGs (passed by build system):

| ARG                 | Example                    | Purpose                      |
| ------------------- | -------------------------- | ---------------------------- |
| `BASE_IMAGE`        | `vllm/vllm-openai:v0.17.1` | Upstream base image          |
| `FRAMEWORK`         | `vllm`                     | Framework name for telemetry |
| `FRAMEWORK_VERSION` | `0.17.1`                   | Version for telemetry        |
| `CONTAINER_TYPE`    | `general`                  | Container type for telemetry |

### Common ARGs with defaults:

| ARG             | Default   | Purpose                       |
| --------------- | --------- | ----------------------------- |
| `PYTHON`        | `python3` | Python binary name            |
| `EFA_VERSION`   | `1.47.0`  | EFA installer version         |
| `CACHE_REFRESH` | `0`       | Cache-busting for apt upgrade |

## 5. Multi-target Build Pattern

Each Dockerfile defines multiple `FROM base AS <target>` stages:

```
vllm Dockerfile:       vllm-ec2, vllm-rayserve-ec2, vllm-sagemaker
sglang Dockerfile:     sglang-ec2, sglang-sagemaker
vllm AL2023:           vllm-ec2-amzn2023, vllm-sagemaker-amzn2023
sglang AL2023:         sglang-ec2, sglang-sagemaker
```

The CI selects the target via `--target` flag in `docker buildx build`.

Each target stage:

1. Runs `apt-get upgrade` (or `dnf upgrade`) with CUDA packages held
1. Copies the appropriate entrypoint script
1. Sets `ENTRYPOINT`

### Entrypoint conventions:

| Framework | EC2                                    | SageMaker                                |
| --------- | -------------------------------------- | ---------------------------------------- |
| vLLM      | `/usr/local/bin/dockerd_entrypoint.sh` | `/usr/local/bin/sagemaker_entrypoint.sh` |
| SGLang    | `/usr/bin/serve`                       | `/usr/bin/serve`                         |

## 6. CI/CD Build Config Pattern

Each image variant has a YAML config in `.github/config/`:

```yaml
image:
  name: "vllm-ec2"
  description: "vLLM for EC2 instances"
common:
  framework: "vllm"
  framework_version: "0.17.1"
  job_type: "general"
  python_version: "py312"
  cuda_version: "cu129"
  os_version: "ubuntu22.04"
  customer_type: "ec2"
  arch_type: "x86"
  prod_image: "vllm:0.17-gpu-py312-ec2"
  device_type: "gpu"
  contributor: "None"
release:
  release: true
  force_release: false
  public_registry: true
  private_registry: true
  enable_soci: true
  environment: production
```

The workflow reads this config, passes values to the `build-image` action, which calls `build_image.sh`.

## 7. SageMaker Labels

SageMaker images get additional OCI labels (applied by `build_image.sh` when `customer_type == "sagemaker"`):

```
com.amazonaws.ml.engines.sagemaker.dlc.arch.<arch>=true
com.amazonaws.ml.engines.sagemaker.dlc.device.<device>=true
com.amazonaws.ml.engines.sagemaker.dlc.framework.<framework>.<version>=true
com.amazonaws.ml.engines.sagemaker.dlc.job.<type>=true
com.amazonaws.ml.engines.sagemaker.dlc.os.<os>=true
com.amazonaws.ml.engines.sagemaker.dlc.python.<python>=true
```

## 8. SageMaker Entrypoint Pattern

Both vLLM and SGLang SageMaker entrypoints follow the same pattern:

- Read env vars with a framework-specific prefix (`SM_VLLM_*` or `SM_SGLANG_*`)
- Convert env var names to CLI args: `SM_VLLM_TENSOR_PARALLEL_SIZE=4` → `--tensor-parallel-size 4`
- Handle boolean flags: `true` → flag only, `false` → skip
- Auto-detect model from `/opt/ml/model` or `HF_MODEL_ID`
- Default port 8080

## 9. Key Observations for New DLC (llama.cpp)

1. **No shared base Dockerfile** — each framework has its own Dockerfile(s) in `docker/<framework>/`
1. **Two variants needed**: Ubuntu overlay (`Dockerfile`) + AL2023 from-source (`Dockerfile.amzn2023`)
1. **Scripts go in** `scripts/<framework>/` with `dockerd_entrypoint.sh` and `sagemaker_entrypoint.sh`
1. **Config goes in** `.github/config/<framework>-<target>.yml`
1. **Workflows go in** `.github/workflows/pr-<framework>-<target>.yml`
1. **The DLC overlay is copy-paste** — telemetry, EFA, OSS compliance, CVE patches are identical
1. **llama.cpp is CPU-capable** — may not need CUDA/EFA sections, making it simpler than vLLM/SGLang
1. **No .dockerignore** for vllm/sglang — only lambda has one
1. **Build context is repo root** (`.`) — all COPY paths are relative to repo root
