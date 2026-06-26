# Problem Set 2: GitHub Actions Workflows

## Background

You are working in the `aws/deep-learning-containers` repository on a test branch. The repository uses GitHub Actions to build and test Docker images. Image variants are defined by YAML config files in `.github/config/image/`, and workflows parse these configs to determine what to build and test.

Your goal is to learn how GitHub Actions workflows work by writing a series of progressively more complex workflows. You'll use the existing infrastructure (config files, custom actions) already in the repo.

## Key Files to Reference

| Path | Purpose |
|------|---------|
| `.github/config/image/pytorch/2.11-ec2-cuda.yml` | Example image config file |
| `.github/actions/discover-configs/` | Action that finds config files matching a pattern |
| `.github/actions/discover-configs/discover_configs.sh` | Script that parses configs into JSON |
| `.github/workflows/pytorch.pr-2.11-cuda.yml` | Example PR workflow (reference only) |
| `.github/workflows/pytorch.pipeline.yml` | Reusable pipeline workflow (reference only) |

## Example Image Config Structure

```yaml
image:
  name: "pytorch-2.11-ec2-cuda"
  description: "..."

metadata:
  framework: "pytorch_runtime"
  framework_version: "2.11.0"
  os_version: "amzn2023"
  customer_type: "ec2"
  device_type: "gpu"
  job_type: "training"
  prod_image: "pytorch:2.11-cu130-amzn2023"

build:
  dockerfile: "docker/pytorch/2.11/Dockerfile.cuda"
  python_version: "3.12"
  cuda_version: "13.0.2"
  torch_version: "2.11.0"

release:
  release: true
  environment: "production"
```

---

## Problem 1: Basic Workflow

Write a workflow file (`.github/workflows/intern-exercise.yml`) that:

- Triggers on `pull_request` events to the `main` branch (types: opened, reopened, synchronize)
- Also triggers on `workflow_dispatch` (so you can test it manually)
- Has a single job that prints:
  - The PR number (or "manual" if triggered via dispatch)
  - The branch name
  - The SHA of the head commit

**Deliverable:** The workflow YAML file and a screenshot of a successful run showing the printed values.

---

## Problem 2: Parse an Image Config

Add a job to your workflow that:

- Reads the file `.github/config/image/pytorch/2.11-ec2-cuda.yml`
- Extracts the following fields using `yq`:
  - `image.name`
  - `metadata.framework`
  - `metadata.framework_version`
  - `metadata.device_type`
  - `metadata.customer_type`
  - `metadata.prod_image`
- Outputs each field as a step output via `$GITHUB_OUTPUT`
- Prints a summary line: `"Image: pytorch-2.11-ec2-cuda | framework=pytorch_runtime | version=2.11.0 | device=gpu | customer=ec2"`

**Deliverable:** The updated workflow file and the run output showing the parsed values.

---

## Problem 3: Conditional Logic

Extend your workflow so that after parsing the config, it:

- Prints "GPU build: CUDA acceleration enabled" if `device_type` is "gpu"
- Prints "CPU build: no accelerator" if `device_type` is "cpu"
- Prints "SageMaker image: adding SM labels" if `customer_type` is "sagemaker"
- Prints "EC2 image: no SM labels needed" if `customer_type` is "ec2"

Use `if:` conditions on steps (not bash if/else) to accomplish this.

**Deliverable:** The updated workflow and a run showing the correct conditional output for the pytorch-2.11-ec2-cuda config.

---

## Problem 4: Multi-Job with Outputs

Refactor your workflow into two jobs:

- **Job 1 (`parse-config`):** Parses the config file and outputs the extracted fields
- **Job 2 (`summarize`):** Depends on Job 1 via `needs:`, consumes the outputs, and prints a formatted summary

Job 2 should not read the config file itself — it should only use the outputs from Job 1.

**Deliverable:** The refactored workflow showing data flowing between jobs via `needs.<job>.outputs.<field>`.

---

## Problem 5: Dynamic Config Discovery

Instead of hardcoding which config file to parse, make your workflow discover which config files were changed in the PR.

- Use `dorny/paths-filter` or `git diff` against the base branch to detect changed files under `.github/config/image/`
- If no config files changed, print "No image config changes detected" and exit cleanly
- If config files changed, print which ones and parse each of them

Hint: Look at how `pytorch.pr-2.11-cuda.yml` uses `dorny/paths-filter` for change detection, and how `discover-configs` action works.

**Deliverable:** The workflow that correctly identifies changed config files on a test PR where you modify one config file.

---

## Problem 6: Matrix Strategy

Extend your workflow to handle multiple config files in parallel using a matrix strategy:

- Use the `discover-configs` action (or your own implementation) to produce a JSON array of config files
- Use `strategy.matrix.include` with `fromJson()` to fan out parallel jobs — one per config file
- Each matrix job parses its own config file and prints the summary

For testing, use the pattern `.github/config/image/pytorch/*.yml` to discover all PyTorch configs.

Expected behavior: if there are 4 PyTorch config files, 4 parallel jobs run, each parsing and printing its own config's metadata.

Hint: Look at how `pytorch.pr-2.11-cuda.yml` uses `needs.discover.outputs.configs` with `fromJson()` to build its matrix.

**Deliverable:** The workflow showing parallel matrix jobs, each printing different image metadata. Screenshot showing multiple parallel jobs in the Actions UI.
