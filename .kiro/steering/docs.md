# Guideline when creating a documentations website for Deep Learning Containers

When working in `deep-learning-containers` repository under the directory `docs`, you are actively working on writing documentations for AWS Deep Learning Containers.

## Project guideline

Our documentations uses Material for MkDocs lightweight framework for all our static documentations. When working, ensure that we are using industry best practices when organizing the codebase unless there are any specific instructions provided.

Within this documentations page, website navigation through `.nav.yml` file should be in one file at `docs/.nav.yml` unless a developer specify otherwise. This is done so that we have a central place to organize our pages.

## Documentation Generation System

The documentation uses an automatic generation system for `available_images.md` and `support_policy.md`.

### Directory Structure

```
docs/src/
├── templates/reference/           # Jinja2 templates
│   ├── available_images.template.md
│   └── support_policy.template.md
├── tables/                        # Table column configs (one per repository)
│   ├── pytorch-training.yml
│   ├── pytorch-inference.yml
│   └── ...
├── data/                          # Image configs (one file per image)
│   ├── pytorch-training/
│   │   ├── 2.9-gpu-ec2.yml
│   │   ├── 2.9-cpu-sagemaker.yml
│   │   └── ...
│   ├── pytorch-training-neuronx/
│   │   └── 2.9-neuronx-sdk2.27.1.yml
│   └── ...
├── legacy/                        # Historical support data
│   └── legacy_support.yml
├── constants.py                   # Path constants and global variables
├── file_loader.py                 # File loading utilities (YAML, configs, templates)
├── image_config.py                # ImageConfig class and image-related functions
├── generate.py                    # Generation logic
├── global.yml                     # Shared terminology and configuration
├── hooks.py                       # MkDocs hooks
├── logger.py                      # Logging configuration
├── macros.py                      # MkDocs macros
├── main.py                        # CLI entry point
└── utils.py                       # Pure utility functions (render_table, parse_version, etc.)
```

### File Responsibilities

- `constants.py` - Path constants and global variables
- `file_loader.py` - All file loading: `load_yaml()`, `load_global_config()`, `load_table_config()`, `load_legacy_support()`, `load_jinja2()`
- `utils.py` - Pure utility functions: `render_table()`, `write_output()`, `parse_version()`, `clone_git_repository()`, `build_ecr_url()`, `build_public_registry_note()`, `check_public_registry()`
- `image_config.py` - `ImageConfig` class, image loaders, sorting functions, row building helpers
- `generate.py` - `generate_support_policy()`, `generate_available_images()`, `generate_all()`
- `macros.py` - MkDocs macros plugin integration
- `hooks.py` - MkDocs hooks entry point

### ImageConfig Class

The `ImageConfig` class provides a dynamic, config-driven interface for image data:

```python
from image_config import ImageConfig, load_repository_images

# Load from YAML file
img = ImageConfig.from_yaml(Path("data/pytorch-training/2.9-gpu-ec2.yml"), "pytorch-training")

# Access any YAML field as attribute
img.version  # "2.9"
img.framework  # "PyTorch"
img.accelerator  # "gpu"
img.repository  # "pytorch-training"

# Safe access with default
img.get("cuda", "-")  # "cu130" or "-" if not present

# Built-in methods
img.is_supported()  # True if eop >= today
img.has_support_dates()  # True if ga and eop fields exist
img.get_display_name(global_config)  # "PyTorch Training"
img.get_framework_group(global_config)  # "pytorch" or None
```

### Image Loading Functions

```python
from image_config import load_image, load_repository_images, load_all_images

# Load single image
img = load_image(Path("data/pytorch-training/2.9-gpu-ec2.yml"), "pytorch-training")

# Load all images for a repository
images = load_repository_images("pytorch-training")  # list[ImageConfig]

# Load all images across all repositories
all_images = load_all_images()  # dict[str, list[ImageConfig]]
```

### Sorting Functions

```python
from image_config import sort_images_for_table, sort_support_entries

# Sort images for available_images.md (version desc, sagemaker first, gpu first)
sorted_images = sort_images_for_table(images)

# Sort support policy entries by table_order then version desc
sorted_entries = sort_support_entries(entries, table_order)
```

### Table Building

```python
from image_config import build_image_row, get_field_display
from utils import render_table

# Get display value for a field (handles platform/accelerator mappings)
value = get_field_display(img, "platform", global_config)  # "EC2, ECS, EKS"

# Build a complete row
row = build_image_row(img, columns, global_config)  # ["PyTorch 2.9", "py312", ...]

# Render markdown table
table = render_table(headers, rows)
```

### Adding a New Image

1. Create a new YAML file in `docs/src/data/<repository>/`:

   - Naming: `<version>-<accelerator>-<platform>.yml`
   - For Neuron: `<version>-neuronx-sdk<sdk_version>.yml`

1. Required fields:

   ```yaml
   framework: PyTorch           # Display name
   version: "2.9"               # Framework version
   accelerator: gpu             # gpu, cpu, or neuronx
   python: py312                # Python version
   platform: ec2                # ec2 or sagemaker
   tag: "2.9.0-gpu-py312-cu130-ubuntu22.04-ec2"  # Full image tag
   ```

1. Optional fields:

   ```yaml
   cuda: cu130                  # For GPU images
   sdk: "2.27.1"                # For Neuron images
   os: ubuntu22.04              # Operating system
   public_registry: true        # If in ECR Public Gallery
   ga: "2025-10-15"             # GA date (for support policy)
   eop: "2026-10-15"            # EOP date (for support policy)
   ecr_account: "007439368137"  # Override default ECR account
   ```

1. GA/EOP dates are only needed for repositories that appear in support policy (base, pytorch-*, tensorflow-* excluding neuronx).

### Adding a New Repository

1. Create directory: `docs/src/data/<repository-name>/`

1. Create table config: `docs/src/tables/<repository-name>.yml`

   ```yaml
   columns:
     - field: framework_version
       header: "Framework"
     - field: python
       header: "Python"
     # ... add columns in desired order
   ```

1. Add to `docs/src/global.yml`:

   - Add display name to `display_names` (required, error if missing)
   - Add to `table_order` list

### Global Configuration (global.yml)

Uses OmegaConf for variable interpolation with `${var}` syntax:

```yaml
# Base terminology
aws: "AWS"
amazon: "Amazon"
dlc: "Deep Learning Containers"

# Composed terminology (resolved at load time)
dlc_long: "${aws} ${dlc}"        # -> "AWS Deep Learning Containers"
sagemaker: "${amazon} SageMaker" # -> "Amazon SageMaker"

# Platform display mappings
platforms:
  ec2: "EC2, ECS, EKS"
  sagemaker: "SageMaker"

# Repository display names (required for all repositories)
display_names:
  pytorch-training: "PyTorch Training"
  # ...

# Framework groups for support policy consolidation (lowercase keys)
framework_groups:
  pytorch:
    - pytorch-training
    - pytorch-inference
    - pytorch-training-arm64
    - pytorch-inference-arm64
  tensorflow:
    - tensorflow-training
    - tensorflow-inference
    - tensorflow-inference-arm64

# Table order (controls order in available_images.md and support_policy.md)
table_order:
  - base
  - pytorch-training
  # ...
```

### Support Policy Consolidation

The `framework_groups` configuration consolidates support policy rows by framework. Repositories in the same group are combined into a single row using the framework name (e.g., "PyTorch").

**Requirements:**

- All repositories in a group that have a given version must have identical GA/EOP dates
- Missing versions in some repositories are allowed (only present repos are consolidated)
- A `ValueError` is raised if dates differ within a group for the same version

To add a new framework group, add an entry to `framework_groups` with the framework name as key and list of repositories as value.

### Reordering Tables and Columns

- **Table order**: Modify `table_order` list in `global.yml`
- **Column order**: Modify `columns` list order in `tables/<repository>.yml`

### Image Sorting in Tables

Images in `available_images.md` are automatically sorted by:

1. **Version** (descending) - newest versions first
1. **Platform** - SageMaker before EC2
1. **Accelerator** - GPU before NeuronX before CPU

### Running Generation

```bash
# Development (use venv in base directory)
cd /path/to/deep-learning-containers
source .venv/bin/activate
cd docs/src && python main.py --verbose

# With MkDocs (automatic via hooks)
mkdocs serve
mkdocs build
```

### Support Policy Logic

- Scans all image configs for `ga` and `eop` fields
- Groups by (framework_group or repository, version)
- Validates consistency within each group
- Auto-determines supported/unsupported by comparing `eop` to current date
- Uses `display_names` for Framework column

### Legacy Support Data

Historical support policy data for older, unsupported images is stored in `docs/src/legacy/legacy_support.yml`.

#### File Format

```yaml
pytorch:
  - version: "2.5"
    ga: "2024-10-29"
    eop: "2025-10-29"
  - version: "2.4"
    ga: "2024-07-24"
    eop: "2025-07-24"
```

#### Adding Legacy Entries

1. Open `docs/src/legacy/legacy_support.yml`
1. Add entries under the framework name key (must match `display_names` keys for framework groups, e.g., `pytorch`, `tensorflow`)
1. Each entry needs: `version`, `ga`, `eop`

#### Behavior

- Legacy entries appear only in `support_policy.md` (unsupported section)
- Images past their EOP date are automatically filtered from `available_images.md`
- The `ImageConfig.is_supported()` method checks if `eop >= today`

## Update knowledge base

If there are any new changes to the documentations generation and organization, make sure to update you knowledge base in the steering/docs.md file and any runbook or update to processes should also be updated in DEVELOPMENT.md files.
This is done so that developers get the most up-to-date information on the current codebase.
