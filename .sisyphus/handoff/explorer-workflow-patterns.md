# CI/CD Workflow Patterns — deep-learning-containers

## 1. All Workflow Files in .github/workflows/

```
# PR workflows (pull_request trigger)
pr-vllm-ec2.yml
pr-vllm-ec2-amzn2023.yml
pr-vllm-sagemaker.yml
pr-vllm-sagemaker-amzn2023.yml
pr-vllm-rayserve.yml
pr-sglang-ec2.yml
pr-sglang-ec2-amzn2023.yml
pr-sglang-sagemaker.yml
pr-sglang-sagemaker-amzn2023.yml
pr-ray-ec2-cpu.yml
pr-ray-ec2-gpu.yml
pr-ray-sagemaker-cpu.yml
pr-ray-sagemaker-gpu.yml
pr-pytorch-ec2.yml
pr-lambda.yml
pr-docs.yml

# Auto-release workflows (schedule + workflow_dispatch)
auto-release-vllm-ec2.yml
auto-release-vllm-sagemaker.yml
auto-release-vllm-rayserve.yml
auto-release-sglang-ec2.yml
auto-release-sglang-sagemaker.yml

# Reusable workflows (workflow_call trigger)
reusable-sanity-tests.yml
reusable-security-tests.yml
reusable-telemetry-tests.yml
reusable-vllm-upstream-tests.yml
reusable-vllm-model-tests.yml
reusable-vllm-sagemaker-tests.yml
reusable-sglang-upstream-tests.yml
reusable-sglang-sagemaker-tests.yml
reusable-release-image.yml

# Other
vllm-heavy-test.yml          # Manual workflow_dispatch for EFA tests
merge.yml
docs.yml
stale.yml
detect-versions.yml
check-upstream-releases.yml
```

## 2. Workflow Architecture Overview

### Standard Job Pipeline (shared by ALL PR and auto-release workflows)

```
gatekeeper → load-config → check-changes → build-image → [test jobs] → [release jobs]
     │             │              │               │
     │        reads YAML      dorny/paths     .github/actions/
     │        config file      -filter         build-image
     │
  .github/actions/
  pr-permission-gate
```

### PR Workflow Pattern (e.g., pr-vllm-ec2.yml)

Every PR workflow follows this exact structure:

1. **gatekeeper** — permission gate from base branch (`.github/actions/pr-permission-gate`)
1. **load-config** — reads `.github/config/<framework>-<platform>.yml`, parses with jq
1. **check-changes** — `dorny/paths-filter@v3` detects which files changed (build, test, etc.)
1. **build-image** — conditional on `build-change == 'true'`, uses `.github/actions/build-image`
1. **Test jobs** — each calls a reusable workflow:
   - `reusable-sanity-tests.yml` — always
   - `reusable-security-tests.yml` — always (on build)
   - `reusable-telemetry-tests.yml` — always
   - Framework-specific upstream tests (reusable-vllm-upstream-tests / reusable-sglang-upstream-tests)
   - Platform-specific tests (reusable-vllm-sagemaker-tests / reusable-sglang-sagemaker-tests)
   - Model smoke tests (reusable-vllm-model-tests — amzn2023 only)

### Auto-Release Workflow Pattern (e.g., auto-release-vllm-ec2.yml)

Same as PR but:

- Triggered by **cron schedule** (Mon/Wed for vLLM, Tue/Thu for SGLang) + `workflow_dispatch`
- No gatekeeper or check-changes jobs
- Always builds (no conditional)
- Adds **generate-release-spec** job after tests pass
- Adds **release-image** job calling `reusable-release-image.yml`
- Release gated by `config.release.release == true`

## 3. Config-Driven Architecture

All workflows read from `.github/config/<name>.yml`. Config files exist for:

```
vllm-ec2.yml                 vllm-sagemaker.yml
vllm-ec2-amzn2023.yml        vllm-sagemaker-amzn2023.yml
vllm-rayserve.yml            vllm-model-tests.yml
sglang-ec2.yml               sglang-sagemaker.yml
sglang-ec2-amzn2023.yml      sglang-sagemaker-amzn2023.yml
```

### Config Structure (example: vllm-ec2.yml)

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

## 4. Composite Actions (in .github/actions/)

| Action                  | Purpose                                                                                                                   |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `build-image`           | Docker build via buildkit, ECR push. Uses `target` arg for multi-stage Dockerfile. Calls `.github/scripts/build_image.sh` |
| `load-config`           | Reads YAML config file, outputs JSON                                                                                      |
| `ecr-authenticate`      | ECR login for pulling/pushing images                                                                                      |
| `pr-permission-gate`    | Checks PR author permissions                                                                                              |
| `generate-release-spec` | Generates release spec YAML from config JSON                                                                              |
| `setup-release-package` | Installs release tooling                                                                                                  |

## 5. How Workflows Reference Dockerfiles, Tests, and Build/Push

### Dockerfiles

- Referenced via `dockerfile-path` input to `build-image` action
- Pattern: `docker/<framework>/Dockerfile` (ubuntu-based) or `docker/<framework>/Dockerfile.amzn2023`
- Multi-stage builds with `target` parameter:
  - vLLM: `vllm-ec2`, `vllm-sagemaker`, `vllm-ec2-amzn2023`, `vllm-sagemaker-amzn2023`, `vllm-rayserve-ec2`
  - SGLang: `sglang-ec2`, `sglang-sagemaker`

### Base Images

- **vLLM (ubuntu)**: `vllm/vllm-openai:v<version>` (upstream vLLM image)
- **vLLM (amzn2023)**: `nvidia/cuda:12.9.1-devel-amzn2023` (builds from scratch)
- **SGLang (ubuntu)**: `lmsysorg/sglang:v<version>-<cuda>-amd64` (upstream SGLang image)
- **SGLang (amzn2023)**: `nvidia/cuda:12.9.1-devel-amzn2023` (builds from scratch)

### Tests

- **Sanity tests**: `test/sanity/` — filesystem checks, labels, OSS compliance, inference engine sanity
- **Security tests**: `test/security/` — ECR vulnerability scanning
- **Telemetry tests**: `test/telemetry/` — telemetry environment and instance tests
- **vLLM upstream tests**: Checks out `vllm-project/vllm` at matching version, runs regression/CUDA/example tests via shell scripts in `scripts/vllm/`
- **SGLang upstream tests**: Checks out `sgl-project/sglang` at matching version, runs benchmark + SRT backend tests
- **SageMaker endpoint tests**: `test/vllm/sagemaker/` and `test/sglang/sagemaker/` — pytest-based
- **vLLM model smoke tests**: Config-driven matrix from `.github/config/vllm-model-tests.yml`, downloads models from S3

### Build/Push

- Build uses `buildkitd` (`.github/scripts/buildkitd.sh` + `.github/scripts/build_image.sh`)
- Images pushed to CI ECR: `<CI_AWS_ACCOUNT_ID>.dkr.ecr.<region>.amazonaws.com/ci:<tag>`
- Release uses `reusable-release-image.yml` which:
  1. Publishes images (with SOCI index generation via nerdctl)
  1. Generates release info (SBOM via inspector-sbomgen)
  1. Publishes notifications
  1. Creates docs PR via GitHub App

### Image URI Fallback Pattern

PR workflows use a smart fallback: if build was skipped (no build changes), tests run against the prod image:

```yaml
image-uri: ${{ needs.build-image.result == 'success' && needs.build-image.outputs.ci-image || format('{0}.dkr.ecr.{1}.amazonaws.com/{2}', vars.PROD_AWS_ACCOUNT_ID, vars.AWS_REGION, needs.load-config.outputs.prod-image) }}
```

## 6. Reusable Workflow Input Signatures

### reusable-sanity-tests.yml

Inputs: image-uri, aws-account-id, aws-region, framework, framework-version, python-version, os-version, cuda-version, customer-type, arch-type, device-type, contributor, container-type, transformers-version

### reusable-security-tests.yml

Inputs: image-uri, aws-account-id, aws-region, framework, framework-version

### reusable-telemetry-tests.yml

Inputs: image-uri, aws-account-id, aws-region, framework, framework-version, container-type

### reusable-vllm-upstream-tests.yml

Inputs: image-uri, aws-account-id, aws-region, framework-version, setup-script, example-test-script
Jobs: regression-test, cuda-test, example-test (all on x86-g6xl-runner GPU fleet)

### reusable-vllm-model-tests.yml

Inputs: image-uri, aws-account-id, aws-region
Jobs: load-models (parses vllm-model-tests.yml), test-model-codebuild-fleet (matrix), test-model-runner-scale-sets (matrix)

### reusable-vllm-sagemaker-tests.yml

Inputs: image-uri only
Jobs: endpoint-test (pytest on default-runner)

### reusable-sglang-upstream-tests.yml

Inputs: image-uri, aws-account-id, aws-region, framework-version, benchmark-start-command, run-srt-backend-test (bool, default true)
Jobs: local-benchmark-test (g6xl), srt-backend-test (g6exl)

### reusable-sglang-sagemaker-tests.yml

Inputs: image-uri only
Jobs: endpoint-test (pytest on default-runner)

### reusable-release-image.yml

Inputs: source-image-uri, release-spec, environment, aws-region, runner-fleet
Jobs: validate-release → step1-publish-images → step2-generate-info → step3-publish-notifications → step4-docs-pr

## 7. Runner Fleets

| Fleet                     | Used For                                                                   |
| ------------------------- | -------------------------------------------------------------------------- |
| `ubuntu-latest`           | Config loading, release spec generation                                    |
| `default-runner`          | Sanity tests, security scans, telemetry, SageMaker endpoint tests, release |
| `x86-build-runner`        | Docker image builds (ubuntu-based)                                         |
| `x86-vllm-build-runner`   | vLLM amzn2023 builds (720min timeout)                                      |
| `x86-sglang-build-runner` | SGLang amzn2023 builds (720min timeout)                                    |
| `x86-g6xl-runner`         | GPU tests (upstream, benchmark)                                            |
| `x86-g6exl-runner`        | SGLang SRT backend tests                                                   |
| `gpu-efa-runners`         | EFA/multi-GPU tests, model smoke tests (runner-scale-sets)                 |

## 8. Key Differences: vLLM vs SGLang

| Aspect              | vLLM                                               | SGLang                                                        |
| ------------------- | -------------------------------------------------- | ------------------------------------------------------------- |
| Upstream tests      | regression + CUDA + example (shell scripts)        | benchmark + SRT backend (sglang.bench_serving + run_suite.py) |
| Test setup          | Parameterized setup-script + example-test-script   | Parameterized benchmark-start-command (docker run)            |
| Model smoke tests   | Yes (amzn2023 only, matrix from config)            | No                                                            |
| RayServe variant    | Yes (pr-vllm-rayserve, auto-release-vllm-rayserve) | No                                                            |
| SageMaker tests     | pytest in test/vllm/sagemaker/                     | pytest in test/sglang/sagemaker/                              |
| Release schedule    | Mon/Wed 10AM PST                                   | Tue/Thu 10AM PST                                              |
| Base image (ubuntu) | vllm/vllm-openai:v<ver>                            | lmsysorg/sglang:v<ver>-<cuda>-amd64                           |

## 9. Pattern Summary for Creating New Framework Workflows

To add a new framework (e.g., llama.cpp), you need:

1. **Config files**: `.github/config/llamacpp-ec2.yml`, `llamacpp-sagemaker.yml`, etc.
1. **Dockerfile**: `docker/llamacpp/Dockerfile` (with multi-stage targets like `llamacpp-ec2`, `llamacpp-sagemaker`)
1. **Scripts**: `scripts/llamacpp/` for test setup and test execution
1. **Tests**: `test/llamacpp/` for framework-specific tests, `test/sanity/` already shared
1. **PR workflows**: `pr-llamacpp-ec2.yml`, `pr-llamacpp-sagemaker.yml` — copy vllm/sglang pattern
1. **Auto-release workflows**: `auto-release-llamacpp-ec2.yml`, etc.
1. **Reusable upstream test workflow**: `reusable-llamacpp-upstream-tests.yml` (if upstream tests exist)
1. **Reusable SageMaker test workflow**: `reusable-llamacpp-sagemaker-tests.yml` (if SageMaker variant)

The following reusable workflows are **already shared** and need no changes:

- `reusable-sanity-tests.yml` (already handles vllm/sglang via framework input)
- `reusable-security-tests.yml`
- `reusable-telemetry-tests.yml`
- `reusable-release-image.yml`

The following composite actions are **already shared**:

- `build-image`, `load-config`, `ecr-authenticate`, `pr-permission-gate`, `generate-release-spec`, `setup-release-package`
