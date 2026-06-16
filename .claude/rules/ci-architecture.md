---
description: CI/CD architecture, image config schema, build lifecycle, and workflow conventions
globs: .github/**/*,scripts/wheels/**/*
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
│   ├── ecr-authenticate/            # ECR login
│   ├── pr-permission-gate/          # PR permission check
│   ├── check-image-exists/          # Probe ECR for existing image
│   ├── download-model/              # Download model from S3 for tests
│   └── setup-release-package/       # Download release tooling from S3
├── config/image/                    # Image configs (one file = one released image)
│   ├── base/
│   ├── pytorch/
│   ├── sglang/
│   ├── vllm/
│   ├── vllm-omni/
│   ├── ray/
│   └── xgboost/
├── scripts/
│   ├── build/                       # Framework-specific build lifecycle hooks
│   │   ├── pytorch/
│   │   │   ├── pre_build.sh         # Hook: fetch wheel cache before build
│   │   │   ├── post_build.sh        # Hook: upload wheel cache after build
│   │   │   └── lib/                 # Utilities called by hooks
│   │   │       ├── fetch_wheels.sh
│   │   │       └── upload_wheels.sh
│   │   ├── vllm/
│   │   │   ├── pre_build.sh         # Hook: fetch wheel + sccache
│   │   │   ├── post_build.sh        # Hook: upload wheel + sccache
│   │   │   └── lib/
│   │   │       ├── fetch_wheels.sh
│   │   │       ├── upload_wheels.sh
│   │   │       ├── sync_sccache.sh
│   │   │       └── source_hash.sh
│   │   └── vllm_omni/ → vllm/      # Symlinks (shares vllm's hooks)
│   └── buildspec-cb-fleet.yml       # CodeBuild fleet runner config
├── workflows/                       # GitHub Actions workflows
│   ├── autorelease-*.yml            # Scheduled image releases
│   ├── pr-*.yml                     # PR validation builds
│   ├── dispatch-*.yml               # Manual triggers (benchmarks, wheels, releases)
│   ├── reusable-*.yml               # Reusable test/release workflows
│   ├── scheduled-*.yml              # Cron jobs (stale issues, ECR allowlists)
│   ├── prcheck-*.yml                # PR meta-checks (pre-commit, merge conditions)
│   └── docs-*.yml                   # Documentation builds
└── archive/                         # Old workflows preserved for reference during refactor

scripts/
├── common/                          # Shared install scripts (COPY'd into Docker images)
├── pytorch/                         # PyTorch install scripts (COPY'd into images)
├── vllm/                            # vLLM install scripts (COPY'd into images)
├── sglang/                          # SGLang install scripts (COPY'd into images)
└── wheels/                          # Standalone wheel compilation tool
    └── pytorch/
        ├── build/                   # Compile PyTorch from source
        └── test/                    # Smoke test compiled wheels

test/
├── test_utils/                      # Shared test infrastructure (imported by tests)
│   ├── aws.py
│   ├── efa_helpers.py               # EFA instance lifecycle (setup/teardown)
│   └── ...
├── efa/                             # EFA integration tests
├── sanity/                          # Sanity checks (labels, filesystem)
├── cuda/                            # CUDA runtime/devel tests
├── vllm/                            # vLLM framework tests
├── sglang/                          # SGLang framework tests
└── pytorch/                         # PyTorch framework tests
```

## Design Axioms

1. **One config file = one released image.** Each YAML in `config/image/{family}/` declares a concrete artifact.
2. **Config `build:` block = single source of truth** for all version pins passed to Docker as `--build-arg`.
3. **Convention over configuration for build hooks.** If `.github/scripts/build/{framework}/pre_build.sh` exists, it runs before docker build. No config field needed.
4. **Actions are self-contained.** Each action bundles its own scripts — no cross-references to other actions' internals.
5. **Workflows are thin orchestrators.** They declare triggers, job graph, and runner fleets. Build logic lives in actions/scripts.
6. **Computed values belong in Dockerfiles.** Things like `SETUPTOOLS_SCM_PRETEND_VERSION` are derived inside the Dockerfile from ARGs passed by the build action.
7. **`scripts/` (repo root) = Docker build artifacts** (COPY'd into images, run inside containers). **`.github/scripts/` = CI orchestration** (run on the CI runner, never inside containers).

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
  # platform: "sagemaker"                 # only for sagemaker/hyperpod variants

build:
  dockerfile: "<path>"                    # e.g., docker/vllm/Dockerfile.amzn2023
  target: "<stage>"                       # Docker multi-stage target (optional)
  # All remaining keys are forwarded as --build-arg KEY=value (uppercased)
  python_version: "<semver>"              # e.g., 3.12, 3.13.12
  cuda_version: "<semver>"                # e.g., 13.0.2 (omit for CPU)
  # ... framework-specific pins

tests:
  <test-name>:
    workflow: "<reusable-workflow-path>"
    with:                                 # optional inputs to the workflow
      <key>: "<value>"

release:
  release: <true|false>
  prod_image: "<repo:tag>"
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
        ├── compute_ci_tag.sh          → CI_TAG
        ├── resolve_build_args.py      → EXTRA_BUILD_ARGS + individual env vars
        ├── pre_build/{framework}.sh   → wheel cache fetch, sccache pull (if exists)
        ├── buildkitd.sh               → BuildKit daemon ready
        ├── ecr-authenticate           → Docker logged into ECR
        ├── docker buildx build        → image built + pushed to ECR
        └── post_build/{framework}.sh  → wheel cache upload, sccache push (if exists)
```

## Conventions

### Script interfaces
- **Python scripts:** use `argparse` with named flags (e.g., `--config-file`)
- **Bash scripts:** use `while [[ $# -gt 0 ]]; case` loop with named flags
- **GitHub Actions outputs:** write to `$GITHUB_OUTPUT` (e.g., `echo "key=value" >> $GITHUB_OUTPUT`)
- **Cross-step state:** write to `$GITHUB_ENV` (e.g., `echo "KEY=value" >> $GITHUB_ENV`)

### Action organization
- If a script is ONLY used by one action → lives inside that action's directory
- If a script is called from multiple places → lives in `.github/scripts/`

### Build hook convention
- `.github/scripts/build/{framework}/pre_build.sh` — runs before docker build
- `.github/scripts/build/{framework}/post_build.sh` — runs after docker build
- `.github/scripts/build/{framework}/lib/` — utility scripts called by hooks (never called directly by workflows)
- If no hook exists for a framework (base, ray, sglang) — nothing runs, no error

### Config file organization
- Configs grouped by family in subdirectories: `config/image/{family}/*.yml`
- Glob patterns are safe within a family: `.github/config/image/base/*.yml`
- Adding a new variant = drop a config file in the family directory

### Workflow organization
Three layers of workflows, named by role:

- `pr-{framework}-{variant}.yml` — thin caller, triggers on PR paths for one image variant
- `autorelease-{framework}-{variant}.yml` — thin caller, scheduled/dispatch trigger for one image variant
- `pipeline-{framework}.yml` — reusable orchestrator, handles build → tests → release for one config
- `reusable-{name}.yml` — single-purpose building block (test suite, release step)

### Workflow hierarchy
```
pr-sglang-ec2.yml (trigger + gatekeeper)
  └── pipeline-sglang.yml (orchestrator)
        ├── build-image action
        ├── reusable-sanity-tests.yml
        ├── reusable-telemetry-tests.yml
        ├── reusable-sglang-upstream-tests.yml
        ├── reusable-sglang-model-tests.yml
        ├── reusable-sglang-sagemaker-tests.yml (gated: sagemaker only)
        └── release (gated: inputs.release)
```

### Workflow patterns
- **Callers are per-variant:** `pr-sglang-ec2.yml` and `pr-sglang-sagemaker.yml` are separate files with scoped path triggers. Shared path changes (Dockerfile) fire both; config-only changes fire one.
- **Pipeline is per-framework:** one `pipeline-sglang.yml` handles all SGLang variants. A `config` job parses the config to gate variant-specific tests (e.g., sagemaker test only runs for sagemaker configs).
- **PR vs release:** identical pipeline call, different `release:` flag. Callers pass `release: false` (PR) or `release: true` (autorelease).
- **Multi-config matrix:** callers use `discover-configs` action to glob a family, matrix over results calling the pipeline per entry.
- **Release gating:** `if: ${{ inputs.release && !failure() && !cancelled() }}` — skipped jobs don't block, failures do.
