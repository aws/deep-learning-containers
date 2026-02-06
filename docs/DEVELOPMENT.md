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
│   ├── data/                                           # Per-image configuration files
│   │   ├── template/
│   │   │   └── image-template.yml                      # Template with all fields documented
│   │   ├── pytorch-training/
│   │   │   └── <version>-<accelerator>-<platform>.yml  # Naming is for organization only
│   │   └── ...
│   ├── legacy/                                         # Historical support data
│   │   └── legacy_support.yml
│   ├── tables/                                         # Table column configurations
│   │   └── <repository>.yml
│   ├── templates/
│   │   ├── reference/                                  # Reference page templates
│   │   └── releasenotes/                               # Release notes templates
│   ├── constants.py                                    # Path constants and GLOBAL_CONFIG
│   ├── generate.py                                     # Generation logic
│   ├── global.yml                                      # Shared terminology and configuration
│   ├── hooks.py                                        # MkDocs hooks
│   ├── image_config.py                                 # ImageConfig class
│   ├── macros.py                                       # MkDocs macros plugin
│   ├── main.py                                         # CLI entry point
│   ├── sorter.py                                       # Sorting tiebreaker functions
│   └── utils.py                                        # Utility functions
├── reference/                                          # Generated reference pages
├── releasenotes/                                       # Generated release notes
└── mkdocs.yml
```

______________________________________________________________________

## Adding a New Image

### Step 1: Create Image Config

Create `docs/src/data/<repository>/<version>-<accelerator>-<platform>.yml`:

```yaml
# Required fields
framework: PyTorch
version: "2.9"
accelerator: gpu              # gpu, cpu, or neuronx
platform: ec2                 # ec2 or sagemaker
tags:
  - "2.9.0-gpu-py312-cu130-ubuntu22.04-ec2"

# Optional metadata
python: py312
cuda: cu130
os: ubuntu22.04
public_registry: true
```

The YAML file name is for organizational purposes only. However, make sure that the image configuration file lives in the correct repository directory.

See `docs/src/data/template/image-template.yml` for all available fields.

### Step 2: Regenerate

```bash
cd docs/src && python main.py --verbose
```

______________________________________________________________________

## Adding Support Policy Dates

Add `ga` and `eop` fields to image configs for repositories that appear in support policy:

```yaml
ga: "2025-10-15"    # General Availability date
eop: "2026-10-15"   # End of Patch date
```

**Version Consolidation:**

- Images with the same major.minor version (e.g., `2.6.0` and `2.6.1`) are consolidated into a single row displayed as `2.6` if they have identical GA/EOP dates
- If patch versions have different GA/EOP dates, each is displayed separately with full version (e.g., `2.6.0`, `2.6.1`) and a warning is logged

**Validation:** All images in the same framework group with the same full version (X.Y.Z) must have identical GA/EOP dates.

______________________________________________________________________

## Adding Release Notes

Add these fields to an image config:

```yaml
announcements:
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

Release notes are generated automatically for images with `announcements` and `packages` fields.

### Adding New Optional Sections

1. Add section to image config under `optional`:

   ```yaml
   optional:
     known_issues:
       - "Issue 1"
     deprecation_notice:
       - "This image will be deprecated..."
   ```

1. Add display name to `global.yml`:

   ```yaml
   display_names:
     deprecation_notice: "Deprecation Notice"
   ```

Sections render in YAML order as bullet lists.

Section headers in optional sections are rendered via the section key.
To format your optional section headers, add a new field in `docs/src/global.yml` under `display_names` section.
Eg: deprecation_notice section will render its header as `## deprecation_notice` unless a formatted string is provided in `docs/src/global.yml`.

______________________________________________________________________

## Adding a New Repository

1. Create directory: `docs/src/data/<repository>/`

1. Create table config `docs/src/tables/<repository>.yml`:

   ```yaml
   columns:
     - field: framework_version
       header: "Framework"
     - field: python
       header: "Python"
     - field: example_url
       header: "Example URL"
   ```

1. Add to `docs/src/global.yml`:

   ```yaml
   display_names:
     my-repo: "My Repository"

   table_order:
     - my-repo
   ```

______________________________________________________________________

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

**Available fields:** `framework_version`, `python`, `cuda`, `sdk`, `accelerator`, `platform`, `os`, `example_url`, `version`, `ga`, `eop`, `framework_group`, `repository`, `release_note_link`
To add additional fields, ensure that the image configuration YAML file contains said field of the same name.
Additionally, if you require the field to be formatted, add an additional attribute in `ImageConfig` class of `display_<field_name>` to grab the formatted field.

______________________________________________________________________

## Legacy Support Data

Historical data for unsupported images in `docs/src/legacy/legacy_support.yml`:

```yaml
pytorch:
  - version: "2.5"
    ga: "2024-10-29"
    eop: "2025-10-29"
```

Generally, this is only required if an image configuration file does not already exist and the image is already past its support.

______________________________________________________________________

## Global Configuration

`docs/src/global.yml` contains:

- **Terminology:** `aws`, `dlc_long`, `sagemaker`, etc.
- **display_names:** Repository and package display names
- **framework_groups:** Support policy consolidation groups
- **table_order:** Order of tables displayed within the documentations website (eg: available_images.md and support_policy.md)
- **platforms/accelerators:** Display mappings

______________________________________________________________________

## Running Generation

```bash
# Help
python main.py --help

# Full generation
python main.py --verbose

# Specific outputs
python main.py --available-images-only
python main.py --support-policy-only
python main.py --release-notes-only

# Preview site
cd docs && mkdocs serve
```

______________________________________________________________________

## Local Documentation Development

### Generation Only (No Server)

Run `main.py` to generate documentation without serving:

```bash
cd docs/src && python main.py --verbose
```

This automatically clones `tutorials/` repository and generates markdown files in `reference/` and `releasenotes/` directories without starting a web server.

### Serving Locally

Use `mkdocs serve` to automatically clone `tutorials/` and generate documentation in `reference/` and `releasenotes/` and serve the website:

```bash
cd docs && mkdocs serve
```

The site is typically available at `http://127.0.0.1:8000/deep-learning-containers/` - check the command output for the actual URL.

### Live Reload

Enable automatic reload on content changes:

```bash
mkdocs serve --livereload
```

**Note:** Live reload only detects changes to:

- Markdown file content
- `.nav.yml` content
- `mkdocs.yml` content

Live reload does **not** detect changes requiring documentation regeneration (e.g., image config YAML files, templates). To regenerate documentation, stop the server (`Ctrl+C`) and rerun `mkdocs serve`.

______________________________________________________________________

## Troubleshooting

| Error | Solution |
| --- | --- |
| "Display name not found" | Add repository to `display_names` in `global.yml` |
| "Inconsistent dates" | Ensure all images in same framework group/version have identical GA/EOP |
| Images not appearing | Check repository is in `table_order` |
| Release notes not generating | Ensure `announcements` and `packages` fields are present |
