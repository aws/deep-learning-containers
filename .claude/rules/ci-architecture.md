---
description: CI/CD architecture, image config schema, build lifecycle, and workflow conventions
paths:
  - ".github/**/*"
  - "scripts/ci/**/*"
---

# CI/CD Architecture

## Directory Structure

```
.github/
├── actions/                         # Composite GitHub Actions (self-contained)
│   ├── build-image/                 # Build + push Docker image from config file
│   │   ├── action.yml
│   │   ├── resolve_build_args.py    # Config build: block → EXTRA_BUILD_ARGS
│   │   ├── compute_ci_tag.sh        # Derive CI tag from config metadata
│   │   └── buildkitd.sh             # Bootstrap BuildKit daemon on runner
│   ├── discover-configs/            # Glob config files → JSON matrix for GHA
│   │   ├── action.yml
│   │   └── discover_configs.sh
│   ├── generate-release-spec/       # Config → release specification YAML
│   │   ├── action.yml
│   │   └── generate_release_spec.sh
│   ├── upload-ecr-allowlists/       # Upload security scan allowlists to S3
│   │   ├── action.yml
│   │   └── upload_ecr_allowlists.py
│   ├── ecr-authenticate/            # ECR login + optional image pull
│   ├── pr-permission-gate/          # PR permission check
│   ├── check-image-exists/          # Probe ECR for existing image
│   ├── download-model/              # Download model from S3 with caching + locking
│   ├── resolve-image-uri/           # Resolve CI or prod image URI from config
│   └── setup-release-package/       # Download release tooling from S3
├── config/
│   ├── image/                       # Image configs (one file = one released image)
│   │   ├── base/
│   │   ├── huggingface-vllm/
│   │   ├── pytorch/
│   │   ├── sglang/
│   │   ├── vllm/
│   │   ├── vllm-omni/
│   │   ├── ray/
│   │   └── xgboost/
│   └── model-tests/                 # Model test matrices (one per framework)
│       ├── sglang-model-tests.yml
│       ├── vllm-model-tests.yml
│       └── vllm-omni-model-tests.yml
├── release-schedule.yml             # Autorelease cron schedule (housekeeping, validated by prcheck)
├── scripts/
│   └── buildspec-cb-fleet.yml       # CodeBuild fleet runner setup (installs uv, yq, jq)
└── workflows/                       # GitHub Actions workflows
    ├── {framework}.pipeline.yml     # Reusable orchestrator per framework
    ├── {framework}.pr-{variant}.yml # PR caller per variant (uses discover-configs matrix)
    ├── {framework}.autorelease-{variant}.yml  # Scheduled release per variant
    ├── {framework}.tests-{suite}.yml          # Framework-specific reusable tests
    ├── {framework}.dispatch-{name}.yml        # Manual dispatch workflows (benchmarks, wheels, releases)
    ├── _reusable.{name}.yml        # Cross-framework reusable tests (sanity, security, telemetry, EFA, release)
    ├── _prcheck.{name}.yml         # PR meta-checks (pre-commit, merge conditions, currency-fix)
    ├── _scheduled.{name}.yml       # Cron jobs (stale issues, ECR allowlists, upstream checks)
    └── docs.{name}.yml             # Documentation builds

scripts/
├── ci/                              # CI orchestration (runs on the runner, never inside containers)
│   ├── build/                       # Framework-specific build lifecycle hooks
│   │   ├── vllm_server/            # Canonical hooks (wheel cache + sccache)
│   │   │   ├── pre_build.sh
│   │   │   ├── post_build.sh
│   │   │   └── lib/
│   │   │       ├── fetch_wheels.sh
│   │   │       ├── upload_wheels.sh
│   │   │       ├── sync_sccache.sh
│   │   │       └── source_hash.sh
│   │   ├── vllm/                   # Per-file symlinks → vllm_server/
│   │   ├── vllm_omni/             # Per-file symlinks → vllm_server/
│   │   ├── pytorch_runtime/       # Wheel cache hooks
│   │   │   ├── pre_build.sh
│   │   │   ├── post_build.sh
│   │   │   └── lib/
│   │   └── xgboost/               # External wheel build hooks
│   ├── autocurrency/               # Version detection, upstream checks, docs PR
│   ├── wheels/                     # Standalone wheel compilation tool
│   │   └── pytorch/
│   │       ├── build/
│   │       └── test/
│   └── parse_model_config.py       # Model test config → GHA matrix JSON
│
└── docker/                          # Docker build artifacts (COPY'd into images, run inside containers)
    ├── common/                     # Shared install scripts (EFA, OSS compliance, Python)
    ├── telemetry/                  # DLC telemetry (deep_learning_container.py, bash_telemetry template)
    ├── huggingface/vllm/           # HF vLLM entrypoints, optimizations
    ├── vllm/                       # vLLM entrypoints, sagemaker_serve, patches
    ├── sglang/                     # SGLang entrypoints
    ├── pytorch/                    # PyTorch entrypoints, SSH config, NCCL
    └── ray/                        # Ray entrypoints, sagemaker_serve

test/
├── test_utils/                      # Shared test infrastructure (imported by tests)
├── sanity/                          # Sanity checks (labels, filesystem, credentials, CUDA)
├── security/                        # ECR vulnerability scan + allowlists
├── telemetry/                       # Telemetry environment + instance tests
├── efa/                             # EFA integration tests
├── pytorch/                         # PyTorch tests (unit, single_gpu, multi_gpu, multi_node)
├── vllm/                            # vLLM tests (upstream, model, sagemaker)
├── vllm-omni/                       # vLLM-Omni tests (model, sagemaker)
├── sglang/                          # SGLang tests (upstream, model, sagemaker)
├── ray/                             # Ray tests (ffmpeg, serve, sagemaker)
└── xgboost/                         # XGBoost tests (unit, integ)
```

## Design Axioms

1. **One config file = one released image.** Each YAML in `config/image/{family}/` declares a concrete artifact.
2. **Config `build:` block = single source of truth** for all version pins passed to Docker as `--build-arg`.
3. **Convention over configuration for build hooks.** If `scripts/ci/build/{framework}/pre_build.sh` exists, it runs before docker build. No config field needed.
4. **Actions are self-contained.** Each action bundles its own scripts — no cross-references to other actions' internals.
5. **Workflows are thin orchestrators.** They declare triggers, job graph, and runner fleets. Build logic lives in actions/scripts.
6. **Computed values belong in Dockerfiles.** Things like `SETUPTOOLS_SCM_PRETEND_VERSION` are derived inside the Dockerfile from ARGs passed by the build action.
7. **`scripts/docker/` = Docker build artifacts** (COPY'd into images, run inside containers). **`scripts/ci/` = CI orchestration** (run on the CI runner, never inside containers).

## Image Config Schema

```yaml
image:
  name: "<unique-image-name>"
  description: "<human-readable description>"

metadata:
  framework: "<framework-name>"           # e.g., vllm_server, pytorch_runtime, ray
  framework_version: "<version>"          # e.g., 0.22.1rc0, 2.11.0
  os_version: "<os>"                      # e.g., amzn2023, ubuntu22.04
  customer_type: "<ec2|sagemaker>"
  arch_type: "x86"
  device_type: "<gpu|cpu>"
  job_type: "<general|training|inference>"
  prod_image: "<repo:tag>"                # Prod ECR destination (used by resolve-image-uri for test fallback)
  # platform: "sagemaker"                 # only for sagemaker/hyperpod variants

build:
  dockerfile: "<path>"                    # e.g., docker/vllm/Dockerfile.amzn2023
  target: "<stage>"                       # Docker multi-stage target (optional)
  # All remaining keys are forwarded as --build-arg KEY=value (uppercased)
  python_version: "<semver>"              # e.g., 3.12, 3.13.12
  cuda_version: "<semver>"                # e.g., 13.0.2 (omit for CPU)
  # ... framework-specific pins

release:
  release: <true|false>
  force_release: <true|false>
  public_registry: <true|false>
  private_registry: <true|false>
  enable_soci: <true|false>
  environment: "<production|preprod|gamma>"
```

### Reserved `build:` keys (not forwarded as --build-arg):
- `dockerfile` — used as `-f <path>` in docker buildx
- `target` — used as `--target <stage>` in docker buildx

### Derived values (not stored in config):
- `ci_tag` — computed at runtime by `compute_ci_tag.sh` from metadata + build fields
- `py312` / `cu130` — derived from `build.python_version` / `build.cuda_version`
- `SETUPTOOLS_SCM_PRETEND_VERSION` — computed inside Dockerfile from `FRAMEWORK_VERSION` ARG

## Build Lifecycle

```
Workflow (thin orchestrator)
  └── build-image action
        ├── compute_ci_tag.sh                    → CI_TAG
        ├── resolve_build_args.py                → EXTRA_BUILD_ARGS + individual env vars
        ├── scripts/ci/build/{framework}/pre_build.sh  → wheel cache fetch, sccache pull (if exists)
        ├── buildkitd.sh                         → BuildKit daemon ready
        ├── ecr-authenticate                     → Docker logged into ECR
        ├── docker buildx build                  → image built + pushed to ECR
        └── scripts/ci/build/{framework}/post_build.sh → wheel cache upload, sccache push (if exists)
```

### Hook-injected build-args
The pre_build hook can set additional env vars via `$GITHUB_ENV` that the build step passes to Docker:
- `USE_PREBUILT_WHEEL` — explicitly passed by build-image action if set
- `EXPORT_TARGETS` — triggers intermediate stage exports (wheel-export, sccache-export)

The build-image action also hardcodes these build-args from metadata:
- `FRAMEWORK` — from `metadata.framework`
- `FRAMEWORK_VERSION` — from `metadata.framework_version`
- `CONTAINER_TYPE` — from `metadata.job_type`
- `CACHE_REFRESH` — current date (busts security patch layer cache)

## Conventions

### Script organization
- **`scripts/ci/`** — runs on the CI runner. Question: "Does this execute during CI?" → put it here.
- **`scripts/docker/`** — COPY'd into Docker images. Question: "Does this run inside the container?" → put it here.
- **Action-bundled scripts** — if ONLY used by one action → lives inside that action's directory.

### Script interfaces
- **Python scripts:** use `argparse` with named flags (e.g., `--config-file`). Fall back to `yq` if `pyyaml` not available.
- **Bash scripts:** use `while [[ $# -gt 0 ]]; case` loop with named flags
- **GitHub Actions outputs:** write to `$GITHUB_OUTPUT` (e.g., `echo "key=value" >> $GITHUB_OUTPUT`)
- **Cross-step state:** write to `$GITHUB_ENV` (e.g., `echo "KEY=value" >> $GITHUB_ENV`)

### Build hook convention
- `scripts/ci/build/{framework}/pre_build.sh` — runs before docker build
- `scripts/ci/build/{framework}/post_build.sh` — runs after docker build
- `scripts/ci/build/{framework}/lib/` — utility scripts called by hooks (never called directly by workflows)
- If no hook exists for a framework (base, ray, sglang) — nothing runs, no error
- Hook directory name matches `metadata.framework` (e.g., `vllm_server`, `vllm_omni`, `pytorch_runtime`)
- Per-file symlinks allow sharing hooks across related frameworks (e.g., `vllm/` → `vllm_server/`, `vllm_omni/` → `vllm_server/`)

### Config file organization
- Configs grouped by family in subdirectories: `config/image/{family}/*.yml`
- Glob patterns are safe within a family: `.github/config/image/base/*.yml`
- Adding a new variant = drop a config file in the family directory

### Workflow naming convention
Dot-namespaced by framework, prefixed by `_` for cross-framework utilities:

- `{framework}.pipeline.yml` — reusable orchestrator per framework
- `{framework}.pr-{variant}.yml` — PR trigger per variant (matrix over discovered configs)
- `{framework}.autorelease-{variant}.yml` — scheduled release per variant
- `{framework}.dispatch-{name}.yml` — manual dispatch (benchmarks, wheels, releases)
- `{framework}.tests-{suite}.yml` — framework-specific reusable tests
- `_reusable.{name}.yml` — cross-framework reusable workflows (sanity, security, telemetry, EFA, release)
- `_prcheck.{name}.yml` — PR meta-checks (pre-commit, merge conditions, currency-fix)
- `_scheduled.{name}.yml` — cron jobs (stale issues, upstream checks, ECR allowlists)

### Workflow hierarchy
```
sglang.pr-amzn2023.yml (trigger + gatekeeper + check-changes + discover-configs)
  └── sglang.pipeline.yml (orchestrator, called per config in matrix)
        ├── build-image action
        ├── _reusable.sanity-tests.yml
        ├── _reusable.security-tests.yml
        ├── _reusable.telemetry-tests.yml
        ├── sglang.tests-upstream.yml
        ├── sglang.tests-model.yml
        ├── sglang.tests-sagemaker.yml (gated: sagemaker only)
        ├── _reusable.efa-tests.yml (optional, per-framework)
        └── release-gate → _reusable.release-image.yml (gated: inputs.release)
```

### Workflow patterns
- **Callers use discover-configs matrix:** all PR and autorelease callers discover configs via glob pattern, then matrix-call the pipeline per config. This means adding a new image variant only requires adding a config file.
- **Pipeline is per-framework:** one `sglang.pipeline.yml` handles all SGLang variants. A `ci-config` job parses the config to gate variant-specific tests (e.g., sagemaker test only runs for sagemaker configs, GPU tests only for GPU images).
- **PR vs release:** identical pipeline call, different `release:` flag. Callers pass `release: false` (PR) or `release: true` (autorelease).
- **Release gating:** pipelines use a `release-gate` job gated on three conditions: `github.ref == 'refs/heads/main'`, caller filename contains `.autorelease` or `.dispatch-release` (via `github.workflow_ref`), and `inputs.release == true`. This prevents accidental releases from PR workflows or non-main branches. The gate then checks `release.release` from config and generates the release spec. The actual release job calls `_reusable.release-image.yml`.
- **Test-only runs:** when build is skipped (only test files changed), test jobs receive empty `image-uri` → `resolve-image-uri` action falls back to prod image from `metadata.prod_image`.
- **Graceful skip:** `check-image-exists` action probes ECR; if image not found (first release scenario), tests skip cleanly.
- **PR change detection:** `dorny/paths-filter` outputs per-test-suite flags. The pipeline receives `run-*-test` booleans. If `build-change` is true, all tests run.

### Concurrency
- Pipeline jobs include `${{ inputs.config-file }}` in concurrency group keys to allow parallel runs of different configs.
- Autorelease callers use `cancel-in-progress: false` (never cancel a release).
- PR callers use `cancel-in-progress: true` for build/test jobs (supersede stale runs).

### Runner fleets
- `x86-build-runner` — CPU builds (PyTorch, Base, Ray, XGBoost, Lambda)
- `x86-vllm-build-runner` — GPU compilation (vLLM, vLLM-Omni)
- `x86-sglang-build-runner` — GPU compilation (SGLang)
- `x86-g6xl-runner` — single-GPU tests (1x L4)
- `x86-g6exl-runner` — multi-GPU tests (4x L4)
- `x86-g6e12xl-runner` — large multi-GPU tests (4x L4, more memory)
- `default-runner` — CPU-only jobs (sanity, telemetry, security, SageMaker endpoint tests)
