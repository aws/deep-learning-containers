# SGLang Workflow Comparison: Ubuntu vs AMZN2023

## 1. Complete File Inventory

### Workflow Files (.github/workflows/)

| File                                  | Lines | Type                                   |
| ------------------------------------- | ----- | -------------------------------------- |
| `pr-sglang-ec2.yml`                   | 238   | PR - Ubuntu EC2                        |
| `pr-sglang-sagemaker.yml`             | 253   | PR - Ubuntu SageMaker                  |
| `pr-sglang-ec2-amzn2023.yml`          | 234   | PR - AMZN2023 EC2                      |
| `pr-sglang-sagemaker-amzn2023.yml`    | 249   | PR - AMZN2023 SageMaker                |
| `auto-release-sglang-ec2.yml`         | 210   | Auto Release - EC2 (Ubuntu only)       |
| `auto-release-sglang-sagemaker.yml`   | 219   | Auto Release - SageMaker (Ubuntu only) |
| `reusable-sglang-upstream-tests.yml`  | 134   | Reusable - upstream tests              |
| `reusable-sglang-sagemaker-tests.yml` | 35    | Reusable - SageMaker endpoint tests    |

### Config Files (.github/config/)

| File                            | Lines |
| ------------------------------- | ----- |
| `sglang-ec2.yml`                | 30    |
| `sglang-sagemaker.yml`          | 30    |
| `sglang-ec2-amzn2023.yml`       | 23    |
| `sglang-sagemaker-amzn2023.yml` | 23    |

### Dockerfiles (docker/sglang/)

| File                  | Used By                                     |
| --------------------- | ------------------------------------------- |
| `Dockerfile`          | Ubuntu workflows (both EC2 and SageMaker)   |
| `Dockerfile.amzn2023` | AMZN2023 workflows (both EC2 and SageMaker) |

______________________________________________________________________

## 2. Ubuntu PR Workflows — Detailed Breakdown

### pr-sglang-ec2.yml

- **Config:** `.github/config/sglang-ec2.yml`
- **Dockerfile:** `docker/sglang/Dockerfile` (via `docker/${{ framework }}/Dockerfile`)
- **Base image:** `lmsysorg/sglang:v<version>-<cuda>-amd64`
- **Build target:** `sglang-ec2`
- **Build fleet:** `x86-build-runner`
- **Path triggers:** `**sglang**`, `!docs/**`, `!**amzn2023**`
- **Jobs (7):**
  1. `gatekeeper` — permission gate
  1. `load-config` — parse config YAML
  1. `check-changes` — detect file changes (build-change, sanity-test-change, telemetry-test-change)
  1. `build-image` — build Docker image
  1. `sanity-test` → `reusable-sanity-tests.yml`
  1. `security-test` → `reusable-security-tests.yml`
  1. `telemetry-test` → `reusable-telemetry-tests.yml`
  1. `upstream-tests` → `reusable-sglang-upstream-tests.yml`
- **Benchmark start command:** Direct CLI args (`--model-path Qwen/Qwen3-0.6B --reasoning-parser qwen3 --host 127.0.0.1 --port 30000`)
- **check-changes build-change paths:**
  - `docker/sglang/**`
  - `scripts/sglang/**`
  - `scripts/common/**`
  - `scripts/telemetry/**`
  - `.github/config/sglang-ec2.yml`

### pr-sglang-sagemaker.yml

- **Config:** `.github/config/sglang-sagemaker.yml`
- **Dockerfile:** `docker/sglang/Dockerfile` (via `docker/${{ framework }}/Dockerfile`)
- **Base image:** `lmsysorg/sglang:v<version>-<cuda>-amd64`
- **Build target:** `sglang-sagemaker`
- **Build fleet:** `x86-build-runner`
- **Path triggers:** `**sglang**`, `!docs/**`, `!**amzn2023**`
- **Jobs (8):**
  1. `gatekeeper`
  1. `load-config`
  1. `check-changes` — detects: build-change, sanity-test-change, telemetry-test-change, **sagemaker-test-change**
  1. `build-image`
  1. `sanity-test` → `reusable-sanity-tests.yml`
  1. `telemetry-test` → `reusable-telemetry-tests.yml`
  1. `security-test` → `reusable-security-tests.yml`
  1. `upstream-tests` → `reusable-sglang-upstream-tests.yml`
  1. **`endpoint-test`** → `reusable-sglang-sagemaker-tests.yml` ← **SageMaker-only test**
- **Benchmark start command:** Uses SM_SGLANG\_\* env vars (`SM_SGLANG_MODEL_PATH`, `SM_SGLANG_REASONING_PARSER`, `SM_SGLANG_HOST`, `SM_SGLANG_PORT`)
- **check-changes build-change paths:** Same as EC2 but references `sglang-sagemaker.yml` config
- **Extra check-changes output:** `sagemaker-test-change` watching `test/sglang/sagemaker/**`

______________________________________________________________________

## 3. AMZN2023 PR Workflows — Detailed Breakdown

### pr-sglang-ec2-amzn2023.yml

- **Config:** `.github/config/sglang-ec2-amzn2023.yml`
- **Dockerfile:** `docker/sglang/Dockerfile.amzn2023` (explicit path)
- **Base image:** `nvidia/cuda:12.9.1-devel-amzn2023` ← **DIFFERENT from Ubuntu**
- **Build target:** `sglang-ec2`
- **Build fleet:** `x86-sglang-build-runner` ← **DIFFERENT fleet from Ubuntu** (Ubuntu uses `x86-build-runner`)
- **timeout-minutes:** `720` ← **AMZN2023 has explicit timeout, Ubuntu does not**
- **Path triggers:** `**sglang**`, `!docs/**` ← **NO `!**amzn2023**` exclusion**
- **Jobs (7):**
  1. `gatekeeper`
  1. `load-config`
  1. `check-changes` — detects: build-change, sanity-test-change, telemetry-test-change
  1. `build-image`
  1. `sanity-test` → `reusable-sanity-tests.yml`
  1. `security-test` → `reusable-security-tests.yml`
  1. `telemetry-test` → `reusable-telemetry-tests.yml`
  1. `upstream-tests` → `reusable-sglang-upstream-tests.yml`
- **Benchmark start command:** Direct CLI args (same pattern as Ubuntu EC2: `--model-path Qwen/Qwen3-0.6B ...`)
- **check-changes build-change paths:**
  - `docker/sglang/Dockerfile.amzn2023` ← **Specific Dockerfile, not `docker/sglang/**`**
  - `scripts/sglang/**`
  - `scripts/common/**`
  - `scripts/telemetry/**`
  - `.github/config/sglang-ec2-amzn2023.yml`

### pr-sglang-sagemaker-amzn2023.yml

- **Config:** `.github/config/sglang-sagemaker-amzn2023.yml`
- **Dockerfile:** `docker/sglang/Dockerfile.amzn2023`
- **Base image:** `nvidia/cuda:12.9.1-devel-amzn2023`
- **Build target:** `sglang-sagemaker`
- **Build fleet:** `x86-sglang-build-runner`
- **timeout-minutes:** `720`
- **Path triggers:** `**sglang**`, `!docs/**` ← **NO `!**amzn2023**` exclusion**
- **Jobs (9):**
  1. `gatekeeper`
  1. `load-config`
  1. `check-changes` — detects: build-change, sanity-test-change, telemetry-test-change, **sagemaker-test-change**
  1. `build-image`
  1. `sanity-test` → `reusable-sanity-tests.yml`
  1. `security-test` → `reusable-security-tests.yml`
  1. `telemetry-test` → `reusable-telemetry-tests.yml`
  1. `upstream-tests` → `reusable-sglang-upstream-tests.yml`
  1. **`endpoint-test`** → `reusable-sglang-sagemaker-tests.yml` ← **SageMaker-only test**
- **Benchmark start command:** Uses SM_SGLANG\_\* env vars (same pattern as Ubuntu SageMaker)
- **Extra check-changes output:** `sagemaker-test-change` watching `test/sglang/sagemaker/**`

______________________________________________________________________

## 4. Auto-Release Workflows (Ubuntu Only — No AMZN2023 equivalents exist)

### auto-release-sglang-ec2.yml

- **Config:** `.github/config/sglang-ec2.yml`
- **Trigger:** Cron `00 17 * * 2,4` (Tue/Thu 10AM PDT) + `workflow_dispatch`
- **Dockerfile:** `docker/sglang/Dockerfile`
- **Base image:** `lmsysorg/sglang:v<version>-<cuda>-amd64`
- **Build fleet:** `x86-build-runner`
- **Jobs (7):**
  1. `load-config`
  1. `build-image`
  1. `sanity-test` → `reusable-sanity-tests.yml`
  1. `telemetry-test` → `reusable-telemetry-tests.yml`
  1. `upstream-tests` → `reusable-sglang-upstream-tests.yml`
  1. `generate-release-spec` — needs: load-config, build-image, sanity-test, upstream-tests
  1. `release-image` → `reusable-release-image.yml`
- **No security-test** (unlike PR workflow)
- **No endpoint-test** (EC2 doesn't have SageMaker tests)
- **Benchmark start command:** Direct CLI args

### auto-release-sglang-sagemaker.yml

- **Config:** `.github/config/sglang-sagemaker.yml`
- **Trigger:** Same cron + `workflow_dispatch`
- **Dockerfile:** `docker/sglang/Dockerfile`
- **Base image:** `lmsysorg/sglang:v<version>-<cuda>-amd64`
- **Build fleet:** `x86-build-runner`
- **Jobs (8):**
  1. `load-config`
  1. `build-image`
  1. `sanity-test` → `reusable-sanity-tests.yml`
  1. `telemetry-test` → `reusable-telemetry-tests.yml`
  1. `upstream-tests` → `reusable-sglang-upstream-tests.yml`
  1. **`endpoint-test`** → `reusable-sglang-sagemaker-tests.yml`
  1. `generate-release-spec` — needs: load-config, build-image, sanity-test, upstream-tests, **endpoint-test**
  1. `release-image` → `reusable-release-image.yml`
- **Benchmark start command:** SM_SGLANG\_\* env vars

______________________________________________________________________

## 5. Reusable Workflows

### reusable-sglang-upstream-tests.yml (134 lines)

- **Inputs:** `image-uri`, `aws-account-id`, `aws-region`, `framework-version`, `benchmark-start-command`
- **Jobs (2):**
  1. `local-benchmark-test` — runs on `x86-g6xl-runner`, downloads ShareGPT dataset, starts container with caller-provided command, runs `sglang.bench_serving`
  1. `srt-backend-test` — runs on `x86-g6exl-runner`, checks out sglang source at matching version, runs `run_suite.py --hw cuda --suite stage-a-test-1`
- **Shared by:** ALL workflows (Ubuntu EC2, Ubuntu SM, AMZN2023 EC2, AMZN2023 SM)

### reusable-sglang-sagemaker-tests.yml (35 lines)

- **Inputs:** `image-uri` only
- **Jobs (1):**
  1. `endpoint-test` — runs on `default-runner`, installs test deps, runs `pytest sglang/sagemaker`
- **Shared by:** All SageMaker workflows (Ubuntu SM, AMZN2023 SM, auto-release SM)

______________________________________________________________________

## 6. Reusable Workflow Sharing Analysis

### Ubuntu workflows: ✅ Share reusable workflows between EC2 and SageMaker

- Both use `reusable-sglang-upstream-tests.yml` (with different `benchmark-start-command`)
- SageMaker additionally uses `reusable-sglang-sagemaker-tests.yml`
- Both share `reusable-sanity-tests.yml`, `reusable-security-tests.yml`, `reusable-telemetry-tests.yml`

### AMZN2023 workflows: ✅ Also share the SAME reusable workflows

- Both AMZN2023 PR workflows call the same `reusable-sglang-upstream-tests.yml`
- AMZN2023 SageMaker also calls `reusable-sglang-sagemaker-tests.yml`
- **No duplication** — AMZN2023 uses the exact same reusable workflows as Ubuntu

______________________________________________________________________

## 7. Config File Differences

| Field                      | Ubuntu EC2                 | Ubuntu SM              | AMZN2023 EC2                        | AMZN2023 SM                     |
| -------------------------- | -------------------------- | ---------------------- | ----------------------------------- | ------------------------------- |
| `os_version`               | `ubuntu24.04`              | `ubuntu24.04`          | `amzn2023`                          | `amzn2023`                      |
| `customer_type`            | `ec2`                      | `sagemaker`            | `ec2`                               | `sagemaker`                     |
| `prod_image`               | `sglang:0.5-gpu-py312-ec2` | `sglang:0.5-gpu-py312` | `sglang:0.5-gpu-py312-amzn2023-ec2` | `sglang:0.5-gpu-py312-amzn2023` |
| `release.release`          | `true`                     | `true`                 | `false`                             | `false`                         |
| `release.public_registry`  | `true`                     | `true`                 | `false`                             | `false`                         |
| `release.private_registry` | `true`                     | `true`                 | `false`                             | `false`                         |
| `release.enable_soci`      | `true`                     | `true`                 | `false`                             | `false`                         |
| `release.environment`      | `production`               | `production`           | `gamma`                             | `gamma`                         |
| `framework_version`        | `0.5.9`                    | `0.5.9`                | `0.5.9`                             | `0.5.9`                         |
| `cuda_version`             | `cu129`                    | `cu129`                | `cu129`                             | `cu129`                         |

______________________________________________________________________

## 8. Path Trigger Analysis

### Ubuntu workflows exclude AMZN2023:

```yaml
# pr-sglang-ec2.yml and pr-sglang-sagemaker.yml
paths:
  - "**sglang**"
  - "!docs/**"
  - "!**amzn2023**"    # ← Excludes amzn2023 files
```

### AMZN2023 workflows do NOT exclude Ubuntu:

```yaml
# pr-sglang-ec2-amzn2023.yml and pr-sglang-sagemaker-amzn2023.yml
paths:
  - "**sglang**"
  - "!docs/**"
  # ← No exclusion of Ubuntu files
```

**Implication:** Changes to Ubuntu Dockerfile (`docker/sglang/Dockerfile`) will trigger AMZN2023 workflows too. Changes to `Dockerfile.amzn2023` will NOT trigger Ubuntu workflows.

### check-changes build-change paths differ:

| Workflow     | Dockerfile watched                                  |
| ------------ | --------------------------------------------------- |
| Ubuntu EC2   | `docker/sglang/**` (all files)                      |
| Ubuntu SM    | `docker/sglang/**` (all files)                      |
| AMZN2023 EC2 | `docker/sglang/Dockerfile.amzn2023` (specific file) |
| AMZN2023 SM  | `docker/sglang/Dockerfile.amzn2023` (specific file) |

______________________________________________________________________

## 9. Autocurrency Tracker

The autocurrency tracker at `.github/config/autocurrency-tracker.yml` (lines 41-65) references **only Ubuntu configs**:

```yaml
sglang:
  github_repo: "sgl-project/sglang"
  tag_prefix: "v"
  config_files:
    - path: ".github/config/sglang-ec2.yml"
      prod_image_template: "sglang:{major}.{minor}-gpu-py312-ec2"
    - path: ".github/config/sglang-sagemaker.yml"
      prod_image_template: "sglang:{major}.{minor}-gpu-py312"
  dockerfiles:
    - path: "docker/sglang/Dockerfile"
      base_image_template: "lmsysorg/sglang:v{version}"
```

**No AMZN2023 configs or Dockerfile.amzn2023 are tracked.**

______________________________________________________________________

## 10. Key Differences Summary for Migration Planning

### Build Differences

| Aspect        | Ubuntu                                | AMZN2023                            |
| ------------- | ------------------------------------- | ----------------------------------- |
| Base image    | `lmsysorg/sglang:v<ver>-<cuda>-amd64` | `nvidia/cuda:12.9.1-devel-amzn2023` |
| Dockerfile    | `docker/sglang/Dockerfile`            | `docker/sglang/Dockerfile.amzn2023` |
| Build fleet   | `x86-build-runner`                    | `x86-sglang-build-runner`           |
| Build timeout | (default)                             | `720 minutes`                       |
| OS            | `ubuntu24.04`                         | `amzn2023`                          |

### Test Coverage Comparison

| Test           | Ubuntu EC2 | Ubuntu SM | AMZN2023 EC2 | AMZN2023 SM |
| -------------- | ---------- | --------- | ------------ | ----------- |
| sanity-test    | ✅         | ✅        | ✅           | ✅          |
| security-test  | ✅         | ✅        | ✅           | ✅          |
| telemetry-test | ✅         | ✅        | ✅           | ✅          |
| upstream-tests | ✅         | ✅        | ✅           | ✅          |
| endpoint-test  | ❌         | ✅        | ❌           | ✅          |

**Test coverage is identical between Ubuntu and AMZN2023.**

### Release Infrastructure

| Aspect                    | Ubuntu       | AMZN2023        |
| ------------------------- | ------------ | --------------- |
| Auto-release workflows    | ✅ (2 files) | ❌ (none exist) |
| Release enabled in config | `true`       | `false`         |
| Autocurrency tracker      | ✅ tracked   | ❌ not tracked  |
| Environment               | `production` | `gamma`         |

### Benchmark Start Command Differences

| Platform  | Ubuntu EC2                           | AMZN2023 EC2                                         |
| --------- | ------------------------------------ | ---------------------------------------------------- |
| EC2       | Direct CLI args (`--model-path ...`) | Direct CLI args (`--model-path ...`) — **identical** |
| SageMaker | SM_SGLANG\_\* env vars               | SM_SGLANG\_\* env vars — **identical**               |

______________________________________________________________________

## 11. What Needs to Change for Complete Swap

### Files to DELETE (Ubuntu-specific):

1. `.github/workflows/pr-sglang-ec2.yml`
1. `.github/workflows/pr-sglang-sagemaker.yml`
1. `.github/config/sglang-ec2.yml`
1. `.github/config/sglang-sagemaker.yml`

### Files to RENAME/REPLACE:

1. `.github/workflows/pr-sglang-ec2-amzn2023.yml` → `.github/workflows/pr-sglang-ec2.yml`
1. `.github/workflows/pr-sglang-sagemaker-amzn2023.yml` → `.github/workflows/pr-sglang-sagemaker.yml`
1. `.github/config/sglang-ec2-amzn2023.yml` → `.github/config/sglang-ec2.yml`
1. `.github/config/sglang-sagemaker-amzn2023.yml` → `.github/config/sglang-sagemaker.yml`

### Files to CREATE:

1. `auto-release-sglang-ec2.yml` — new version pointing to AMZN2023 config/Dockerfile
1. `auto-release-sglang-sagemaker.yml` — new version pointing to AMZN2023 config/Dockerfile

### Files to UPDATE:

1. **Autocurrency tracker** (`.github/config/autocurrency-tracker.yml`):
   - Update `dockerfiles[0].path` → `docker/sglang/Dockerfile.amzn2023`
   - Update `dockerfiles[0].base_image_template` → `nvidia/cuda:{version}-devel-amzn2023` (or appropriate pattern)
   - Update `prod_image_template` values if prod_image naming changes
1. **Config files** (after rename):
   - Set `release.release: true`
   - Set `release.environment: production`
   - Set `release.public_registry: true`, `private_registry: true`, `enable_soci: true`
   - Update `prod_image` to drop `-amzn2023` suffix (to match current Ubuntu naming)
1. **Renamed PR workflows**:
   - Remove `-amzn2023` from workflow `name:` field
   - Update `CONFIG_FILE` to point to non-amzn2023 config paths
   - Update path triggers: add `!**amzn2023**` exclusion (or remove if Ubuntu files are deleted)
   - Consider whether build fleet should stay `x86-sglang-build-runner` or revert to `x86-build-runner`

### Files that can stay as-is:

1. `reusable-sglang-upstream-tests.yml` — already shared, no changes needed
1. `reusable-sglang-sagemaker-tests.yml` — already shared, no changes needed
1. `docker/sglang/Dockerfile.amzn2023` — the actual Dockerfile stays

### Files to potentially DELETE (post-swap cleanup):

1. `docker/sglang/Dockerfile` — Ubuntu Dockerfile no longer needed
