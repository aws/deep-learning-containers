# Documentation Development Guide

This guide explains how to use and extend the AWS Deep Learning Containers documentation generation system.

## Overview

The documentation system uses per-image YAML configuration files as the single source of truth to generate:

1. `reference/support_policy.md` - Framework support policy table with GA/EOP dates
1. `reference/available_images.md` - Available images tables grouped by repository
1. `releasenotes/<framework>/*.md` - Individual release notes for each image

## Directory Structure

```
docs/
├── src/
│   ├── data/
│   │   └── releases/           # Configuration files organized by ECR repository
│   │       ├── pytorch-training/
│   │       ├── pytorch-inference/
│   │       ├── tensorflow-training/
│   │       └── ...
│   ├── templates/              # Jinja2 templates
│   │   ├── support_policy.template.md
│   │   ├── available_images.template.md
│   │   └── release_notes.template.md
│   ├── generate.py             # Main generation script
│   ├── config_loader.py        # Config loading and validation
│   └── constants.py            # Configuration constants
├── reference/                  # Generated reference docs
├── releasenotes/              # Generated release notes
└── DEVELOPMENT.md             # This file
```

## How to Add a New Image Release

1. Create a new YAML config file in the appropriate repository directory:

```bash
# Example: Adding PyTorch 3.0 GPU training for SageMaker
touch docs/src/data/releases/pytorch-training/3.0-gpu-sagemaker.yml
```

2. Fill in the configuration (see [Config File Schema](#config-file-schema) below):

```yaml
# PyTorch 3.0 GPU Training on SageMaker
metadata:
  framework: pytorch
  job_type: training
  version: "3.0"
  accelerator: gpu
  platform: sagemaker
  architecture: x86_64
  ga_date: 2026-01-15
  eop_date: 2027-01-15

environment:
  python: "3.12"
  os: ubuntu24.04

packages:
  cuda: "14.0.0"
  cudnn: "9.15.0"
  nccl: "2.28.0"

image:
  public_registry: true
  tags:
    - "3.0.0-gpu-py312-cu140-ubuntu24.04-sagemaker"
```

3. Regenerate documentation:

```bash
cd docs
source .venv/bin/activate
python src/generate.py
```

4. Verify the generated files:
   - `reference/support_policy.md` - Should include the new version
   - `reference/available_images.md` - Should include the new image
   - `releasenotes/pytorch/3.0-gpu-training-sagemaker.md` - New release notes file

## How to Add a New Custom Section Type

Custom sections (like "Known Issues" or "Deprecations") can be added to release notes.

1. Edit `docs/src/constants.py`:

```python
# Add the new section key to ALLOWED_SECTIONS
ALLOWED_SECTIONS = [
    "known_issues",
    "deprecations",
    "breaking_changes",  # Add new section here
]

# Add the display title
SECTION_TITLES = {
    "known_issues": "Known Issues",
    "deprecations": "Deprecations",
    "breaking_changes": "Breaking Changes",  # Add title here
}
```

2. Use the new section in config files:

```yaml
sections:
  breaking_changes:
    - "API X has been removed"
    - "Default behavior of Y has changed"
```

3. Regenerate documentation.

## How to Regenerate Documentation

```bash
cd docs
source .venv/bin/activate
python src/generate.py
```

This will:

- Load all config files from `src/data/releases/`
- Validate each config
- Generate `support_policy.md`
- Generate `available_images.md`
- Generate release notes for each config

## Config File Schema

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `metadata.framework` | string | Framework name: `pytorch`, `tensorflow`, `vllm`, `sglang`, `base` |
| `metadata.job_type` | string | Job type: `training`, `inference`, `base` |
| `metadata.version` | string | Framework version (e.g., `"2.9"`, `"0.13"`) |
| `metadata.accelerator` | string | Accelerator type: `cpu`, `gpu` |
| `metadata.platform` | string | Platform: `sagemaker`, `ec2` |
| `metadata.architecture` | string | Architecture: `x86_64`, `arm64` |
| `metadata.ga_date` | date | General Availability date (YYYY-MM-DD) |
| `metadata.eop_date` | date | End of Patch support date (YYYY-MM-DD) |
| `environment.python` | string | Python version (e.g., `"3.12"`) |
| `environment.os` | string | OS identifier (e.g., `ubuntu22.04`) |
| `image.tags` | list | List of image tags (first tag used in example URLs) |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `packages` | dict | Key-value pairs of package names and versions |
| `image.public_registry` | bool | Whether image is in ECR Public Gallery |
| `sections` | dict | Custom sections (see ALLOWED_SECTIONS) |

### Example Config File

```yaml
# Full example with all fields
metadata:
  framework: pytorch
  job_type: training
  version: "2.9"
  accelerator: gpu
  platform: sagemaker
  architecture: x86_64
  ga_date: 2025-10-15
  eop_date: 2026-10-15

environment:
  python: "3.12"
  os: ubuntu22.04

packages:
  cuda: "13.0.0"
  cudnn: "9.13.0.50"
  nccl: "2.27.7-1"
  efa: "1.43.3"
  transformer_engine: "2.9"
  flash_attention: "2.8.3"

image:
  public_registry: true
  tags:
    - "2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker"

sections:
  known_issues:
    - "Issue description here"
```

## File Naming Conventions

### Config Files

Format: `<version>-<accelerator>-<platform>.yml`

Examples:

- `2.9-gpu-sagemaker.yml`
- `2.9-cpu-ec2.yml`
- `0.13-gpu-sagemaker.yml`

### Generated Release Notes

Format: `<version>-<accelerator>-<job_type>[-arm64]-<platform>.md`

Examples:

- `2.9-gpu-training-sagemaker.md`
- `2.6-cpu-inference-ec2.md`
- `2.6-gpu-inference-arm64-sagemaker.md`

## Adding a New Repository

1. Create the directory:

```bash
mkdir docs/src/data/releases/new-repository
```

2. Add to `docs/src/constants.py`:

```python
REPOSITORY_NAMES = {
    # ... existing entries
    "new-repository": "New Repository Display Name",
}

REPOSITORY_ORDER = [
    # ... existing entries
    "new-repository",
]
```

3. Create config files and regenerate.

## Troubleshooting

### Validation Errors

If you see validation errors like:

```
ValueError: path/to/config.yml: Missing required fields: ['metadata.ga_date']
```

Check that all required fields are present in your config file.

### Template Errors

If templates fail to render, check:

- Jinja2 syntax in `src/templates/*.template.md`
- Variable names match what's passed in `generate.py`

### Missing Dependencies

```bash
cd docs
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```
