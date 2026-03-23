# SGLang CI/CD Workflow Analysis

## 1. File Inventory

### AMZN2023 Workflows (2 files)

| File                                                 | Purpose                           |
| ---------------------------------------------------- | --------------------------------- |
| `.github/workflows/pr-sglang-ec2-amzn2023.yml`       | PR CI for SGLang EC2 AL2023       |
| `.github/workflows/pr-sglang-sagemaker-amzn2023.yml` | PR CI for SGLang SageMaker AL2023 |

### Original Ubuntu Workflows (2 PR + 2 auto-release + 2 reusable = 6 files)

| File                                                    | Purpose                                 |
| ------------------------------------------------------- | --------------------------------------- |
| `.github/workflows/pr-sglang-ec2.yml`                   | PR CI for SGLang EC2 (Ubuntu)           |
| `.github/workflows/pr-sglang-sagemaker.yml`             | PR CI for SGLang SageMaker (Ubuntu)     |
| `.github/workflows/auto-release-sglang-ec2.yml`         | Scheduled release for SGLang EC2        |
| `.github/workflows/auto-release-sglang-sagemaker.yml`   | Scheduled release for SGLang SageMaker  |
| `.github/workflows/reusable-sglang-upstream-tests.yml`  | Reusable: benchmark + srt-backend tests |
| `.github/workflows/reusable-sglang-sagemaker-tests.yml` | Reusable: SageMaker endpoint tests      |

### Config Files (4 files)

| File                                           | Purpose                           |
| ---------------------------------------------- | --------------------------------- |
| `.github/config/sglang-ec2.yml`                | Config for Ubuntu EC2 image       |
| `.github/config/sglang-sagemaker.yml`          | Config for Ubuntu SageMaker image |
| `.github/config/sglang-ec2-amzn2023.yml`       | Config for AL2023 EC2 image       |
| `.github/config/sglang-sagemaker-amzn2023.yml` | Config for AL2023 SageMaker image |

______________________________________________________________________

## 2. AMZN2023 PR Workflow Details

### pr-sglang-ec2-amzn2023.yml

- **Trigger**: `pull_request` on `main`, paths: `**sglang**`, excludes `docs/**`
- **⚠️ Does NOT exclude `!**amzn2023**`** — triggers on ALL sglang changes (overlaps with Ubuntu workflows)
- **Config**: `.github/config/sglang-ec2-amzn2023.yml`
- **Base image**: `nvidia/cuda:12.9.1-devel-amzn2023` (builds from source, not upstream sglang image)
- **Dockerfile**: `docker/sglang/Dockerfile.amzn2023`
- **Build runner**: `x86-sglang-build-runner` (different from Ubuntu's `x86-build-runner`)
- **Build timeout**: 720 minutes (12 hours!) — reflects source compilation
- **Change detection watches**: `docker/sglang/Dockerfile.amzn2023`, `scripts/sglang/**`, `scripts/common/**`, `scripts/telemetry/**`, `.github/config/sglang-ec2-amzn2023.yml`
- **Active jobs**: gatekeeper, load-config, check-changes, build-image, sanity-test, telemetry-test, upstream-tests
- **Commented-out jobs**:
  - `security-test` — TODO comment: "Re-enable ECR scan once AL2023 image vulnerability baseline is established"
- **Benchmark start command**: Uses `SM_SGLANG_*` env vars (same as SageMaker pattern, not direct CLI args)

### pr-sglang-sagemaker-amzn2023.yml

- **Trigger**: Same as EC2 amzn2023 — `**sglang**`, excludes `docs/**`, no amzn2023 exclusion
- **Config**: `.github/config/sglang-sagemaker-amzn2023.yml`
- **Base image**: `nvidia/cuda:12.9.1-devel-amzn2023`
- **Dockerfile**: `docker/sglang/Dockerfile.amzn2023`
- **Build runner**: `x86-sglang-build-runner`, timeout 720 min
- **Change detection watches**: Same as EC2 amzn2023 + `test/sglang/sagemaker/**`
- **Active jobs**: gatekeeper, load-config, check-changes, build-image, sanity-test, telemetry-test, upstream-tests, endpoint-test
- **Commented-out jobs**:
  - `security-test` — Same TODO as EC2 amzn2023
- **Extra vs EC2**: Has `endpoint-test` job (calls `reusable-sglang-sagemaker-tests.yml`)

______________________________________________________________________

## 3. Config File Comparison

| Field             | EC2 Ubuntu               | EC2 AL2023               | SM Ubuntu            | SM AL2023                     |
| ----------------- | ------------------------ | ------------------------ | -------------------- | ----------------------------- |
| os_version        | ubuntu24.04              | amzn2023                 | ubuntu24.04          | amzn2023                      |
| framework_version | 0.5.9                    | 0.5.9                    | 0.5.9                | 0.5.9                         |
| cuda_version      | cu129                    | cu129                    | cu129                | cu129                         |
| python_version    | py312                    | py312                    | py312                | py312                         |
| prod_image        | sglang:0.5-gpu-py312-ec2 | sglang:0.5-gpu-py312-ec2 | sglang:0.5-gpu-py312 | sglang:0.5-gpu-py312-amzn2023 |
| release           | true                     | true                     | true                 | true                          |
| public_registry   | true                     | true                     | true                 | true                          |
| enable_soci       | true                     | true                     | true                 | true                          |

**⚠️ Notable**: EC2 AL2023 `prod_image` is `sglang:0.5-gpu-py312-ec2` — same as Ubuntu EC2. This could cause a collision if both are released. SM AL2023 correctly uses a distinct tag `sglang:0.5-gpu-py312-amzn2023`.

______________________________________________________________________

## 4. Original Ubuntu Workflow Path Exclusions

Both `pr-sglang-ec2.yml` and `pr-sglang-sagemaker.yml` have:

```yaml
paths:
  - "**sglang**"
  - "!docs/**"
  - "!**amzn2023**"    # ✅ Excludes amzn2023 files
```

This means Ubuntu workflows correctly skip when only amzn2023 files change. However, the amzn2023 workflows do NOT have a reciprocal exclusion — they trigger on ALL sglang changes, meaning both Ubuntu and amzn2023 workflows fire together on non-amzn2023-specific changes.

______________________________________________________________________

## 5. Auto-Release Workflows

### Existing (Ubuntu only)

- `auto-release-sglang-ec2.yml` — Cron: Tue/Thu 10AM PDT, config: `sglang-ec2.yml`
- `auto-release-sglang-sagemaker.yml` — Cron: Tue/Thu 10AM PDT, config: `sglang-sagemaker.yml`

Both have commented-out PR triggers and support `workflow_dispatch`.

### ❌ Missing: No auto-release workflows for AMZN2023

There are NO `auto-release-sglang-ec2-amzn2023.yml` or `auto-release-sglang-sagemaker-amzn2023.yml` files.

### Autocurrency Tracker

`.github/config/autocurrency-tracker.yml` tracks sglang but only references Ubuntu configs:

- `.github/config/sglang-ec2.yml`
- `.github/config/sglang-sagemaker.yml`
- `docker/sglang/Dockerfile` (not Dockerfile.amzn2023)

______________________________________________________________________

## 6. Key Findings & Issues

1. **No auto-release for AMZN2023** — PR workflows exist but no scheduled release pipelines
1. **Security tests disabled** — Both amzn2023 PR workflows have `security-test` commented out pending vulnerability baseline
1. **Path trigger overlap** — AMZN2023 workflows trigger on ALL sglang changes (no exclusion for Ubuntu-only files), causing unnecessary CI runs
1. **EC2 AL2023 prod_image collision risk** — `sglang:0.5-gpu-py312-ec2` is identical to Ubuntu EC2 prod_image tag
1. **Autocurrency tracker missing AL2023** — Only tracks Ubuntu configs/Dockerfile, not amzn2023 variants
1. **Build time** — AL2023 images use 720-min timeout (source build) vs Ubuntu's default (uses pre-built upstream image)
1. **Reusable workflows shared** — Both Ubuntu and AL2023 use the same `reusable-sglang-upstream-tests.yml` and `reusable-sglang-sagemaker-tests.yml`
