# Documentation Development Guide

Guide for developers working with AWS Deep Learning Containers documentation.

## Quick Start

```bash
cd /path/to/deep-learning-containers
source .venv/bin/activate
cd docs/src && python main.py --verbose
```

## Directory Structure

```
docs/
├── src/
│   ├── data/                    # Per-image configuration files
│   │   ├── template/
│   │   │   └── image-template.yml  # Template with all fields documented
│   │   ├── pytorch-training/
│   │   │   └── <version>-<accelerator>-<platform>.yml
│   │   └── ...
│   ├── legacy/                  # Historical support data
│   │   └── legacy_support.yml
│   ├── tables/                  # Table column configurations
│   │   └── <repository>.yml
│   ├── templates/
│   │   ├── reference/           # Reference page templates
│   │   └── releasenotes/        # Release notes templates
│   ├── constants.py             # Path constants and GLOBAL_CONFIG
│   ├── generate.py              # Generation logic
│   ├── global.yml               # Shared terminology and configuration
│   ├── hooks.py                 # MkDocs hooks
│   ├── image_config.py          # ImageConfig class
│   ├── macros.py                # MkDocs macros plugin
│   ├── main.py                  # CLI entry point
│   ├── sorter.py                # Sorting tiebreaker functions
│   └── utils.py                 # Utility functions
├── reference/                   # Generated reference pages
├── releasenotes/                # Generated release notes
└── mkdocs.yml
```

* * *

## Adding a New Image

### Step 1: Create Image Config

Create `docs/src/data/<repository>/<version>-<accelerator>-<platform>.yml`:

```yaml
# Required fields
framework: PyTorch
version: "2.9"
accelerator: gpu              # gpu, cpu, or neuronx
python: py312
platform: ec2                 # ec2 or sagemaker
tags:
  - "2.9.0-gpu-py312-cu130-ubuntu22.04-ec2"

# Optional metadata
cuda: cu130
os: ubuntu22.04
public_registry: true
```

See `docs/src/data/template/image-template.yml` for all available fields.

### Step 2: Regenerate

```bash
cd docs/src && python main.py --verbose
```

* * *

## Adding Support Policy Dates

Add `ga` and `eop` fields to image configs for repositories that appear in support policy:

```yaml
ga: "2025-10-15"    # General Availability date
eop: "2026-10-15"   # End of Patch date
```

**Validation:** All images in the same framework group with the same version must have identical GA/EOP dates.

* * *

## Adding Release Notes

Add these fields to an image config:

```yaml
announcement:
  - "Introduced containers for PyTorch 2.9"
  - "Added Python 3.12 support"

packages:
  python: "3.12"
  pytorch: "2.9.0"
  cuda: "13.0"

# Optional sections (rendered dynamically)
optional:
  known_issues:
    - "Description of known issue"
```

Release notes are generated automatically for images with `announcement` and `packages` fields.

### Adding New Optional Sections

1. Add section to image config under `optional`:
   ```yaml
   optional:
     known_issues:
       - "Issue 1"
     deprecation_notice:
       - "This image will be deprecated..."
   ```

2. Add display name to `global.yml`:
   ```yaml
   display_names:
     deprecation_notice: "Deprecation Notice"
   ```

Sections render in YAML order as bullet lists.

* * *

## Adding a New Repository

1. Create directory: `docs/src/data/<repository>/`

2. Create table config `docs/src/tables/<repository>.yml`:
   ```yaml
   columns:
     - field: framework_version
       header: "Framework"
     - field: python
       header: "Python"
     - field: example_url
       header: "Example URL"
   ```

3. Add to `docs/src/global.yml`:
   ```yaml
   display_names:
     my-repo: "My Repository"

   table_order:
     - my-repo
   ```

* * *

## Editing Table Columns

Edit `docs/src/tables/<repository>.yml`:

```yaml
columns:
  - field: framework_version
    header: "Framework"
  - field: python
    header: "Python"
  # Add/remove/reorder columns here
```

**Available fields:** `framework_version`, `python`, `cuda`, `sdk`, `accelerator`, `platform`, `os`, `example_url`, `tag`, `release_note_link`

* * *

## Legacy Support Data

Historical data for unsupported images in `docs/src/legacy/legacy_support.yml`:

```yaml
pytorch:
  - version: "2.5"
    ga: "2024-10-29"
    eop: "2025-10-29"
```

* * *

## Global Configuration

`docs/src/global.yml` contains:

- **Terminology:** `aws`, `dlc_long`, `sagemaker`, etc.
- **display_names:** Repository and package display names
- **framework_groups:** Support policy consolidation groups
- **table_order:** Order of tables in available_images.md
- **platforms/accelerators:** Display mappings

* * *

## Running Generation

```bash
# Full generation
python main.py --verbose

# Specific outputs
python main.py --available-images-only
python main.py --support-policy-only
python main.py --release-notes-only

# Preview site
cd docs && mkdocs serve
```

* * *

## Troubleshooting

| Error | Solution |
| --- | --- |
| "Display name not found" | Add repository to `display_names` in `global.yml` |
| "Inconsistent dates" | Ensure all images in same framework group/version have identical GA/EOP |
| Images not appearing | Check repository is in `table_order` |
| Release notes not generating | Ensure `announcement` and `packages` fields are present |
