# Guideline when creating a documentations website for Deep Learning Containers

When working in `deep-learning-containers` repository under the directory `docs`, you are actively working on writing documentations for AWS Deep Learning Containers.

## Project guideline

Our documentations uses Material for MkDocs lightweight framework for all our static documentations. When working, ensure that we are using industry best practices when organizing the codebase unless there are any specific instructions provided.

Within this documentations page, website navigation through `.nav.yml` file should be in one file at `docs/.nav.yml` unless a developer specify otherwise. This is done so that we have a central place to organize our pages.

## Documentation Generation System

The documentation uses an automatic generation system for `available_images.md`, `support_policy.md`, and release notes.

### Directory Structure

```
docs/src/
├── templates/
│   ├── reference/                 # Reference page templates
│   │   ├── available_images.template.md
│   │   └── support_policy.template.md
│   └── releasenotes/              # Release notes templates
│       ├── release_note.template.md
│       └── index.template.md
├── tables/                        # Table column configs (one per repository)
│   ├── extra/                     # Special table configs for generated pages
│   │   ├── support_policy.yml     # Columns for support_policy.md
│   │   └── release_notes.yml      # Columns for release notes index pages
│   ├── pytorch-training.yml
│   ├── pytorch-inference.yml
│   └── ...
├── data/                          # Image configs (one file per image)
│   ├── template/
│   │   └── image-template.yml     # Template with all possible fields documented
│   ├── pytorch-training/
│   │   ├── 2.9-gpu-ec2.yml
│   │   ├── 2.9-cpu-sagemaker.yml
│   │   └── ...
│   ├── pytorch-training-neuronx/
│   │   └── 2.9-neuronx-sdk2.27.1.yml
│   └── ...
├── legacy/                        # Historical support data
│   └── legacy_support.yml
├── constants.py                   # Path constants, global variables, and GLOBAL_CONFIG
├── generate.py                    # Generation logic
├── global.yml                     # Shared terminology and configuration
├── hooks.py                       # MkDocs hooks
├── image_config.py                # ImageConfig class and image-related functions
├── logger.py                      # Logging configuration
├── macros.py                      # MkDocs macros
├── main.py                        # CLI entry point
├── sorter.py                      # Sorting tiebreaker functions
└── utils.py                       # Utility functions (file loading, table rendering, etc.)
```

### File Responsibilities

- `constants.py` - Path constants, global variables, `GLOBAL_CONFIG`, and `RELEASE_NOTES_REQUIRED_FIELDS`
- `sorter.py` - Sorting tiebreaker functions: `platform_sorter`, `accelerator_sorter`, `repository_sorter`
- `utils.py` - Utility functions: `load_yaml()`, `load_table_config()`, `load_jinja2()`, `render_table()`, `write_output()`, `parse_version()`, `clone_git_repository()`, `build_ecr_uri()`, `build_public_ecr_uri()`, `get_framework_order()`
- `image_config.py` - `ImageConfig` class, image loaders (`load_repository_images`, `load_legacy_images`, `load_images_by_framework_group`), `sort_by_version`, `get_latest_image_uri`, `build_image_row`, `check_public_registry`
- `generate.py` - `generate_support_policy()`, `generate_available_images()`, `generate_release_notes()`, `generate_all()`
- `macros.py` - MkDocs macros plugin integration
- `hooks.py` - MkDocs hooks entry point

### Module Import Structure

The modules follow a strict import hierarchy to avoid circular imports:

```
constants.py  (no project imports)
     ↓
utils.py      (imports from constants)
     ↓
image_config.py (imports from constants and utils)
```

Functions that instantiate `ImageConfig` must stay in `image_config.py`. Functions that only use `GLOBAL_CONFIG` or utilities can live in `utils.py`.

### Global Configuration

The global configuration is loaded once at module import time in `constants.py` and exposed as `GLOBAL_CONFIG`. This eliminates the need to pass `global_config` to most functions.

```python
from constants import GLOBAL_CONFIG

# Access any config value
table_order = GLOBAL_CONFIG.get("table_order", [])
display_names = GLOBAL_CONFIG["display_names"]
```

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
img.framework_group  # "pytorch" (computed from GLOBAL_CONFIG, defaults to repository)

# Safe access with default
img.get("cuda", "-")  # "cu130" or "-" if not present

# Properties (no parentheses)
img.is_supported  # True if eop >= today
img.has_support_dates  # True if ga and eop fields exist
img.has_release_notes  # True if all RELEASE_NOTES_REQUIRED_FIELDS are present
img.display_repository  # "PyTorch Training" (uses GLOBAL_CONFIG)
img.display_framework_group  # "PyTorch" (uses GLOBAL_CONFIG)
img.display_tag  # First tag from tags list (used in available_images.md)
img.release_note_filename  # "<repo>-<version>-<accelerator>-<platform>.md"

# Display properties for table rendering
img.display_framework_version  # "PyTorch 2.9"
img.display_example_url  # formatted ECR URL (uses display_tag)
img.display_platform  # "EC2, ECS, EKS" (mapped from "ec2")
img.display_accelerator  # "GPU" (mapped from "gpu")

# Methods
img.get_image_uris()  # List of image URIs (private ECR + public ECR if public_registry=true)

# Generic display accessor (uses display_<field> property if exists, else raw value)
img.get_display("platform")  # "EC2, ECS, EKS"
img.get_display("python")  # "py312" (no display_ property, returns raw)
```

### Image Loading Functions

```python
from image_config import load_repository_images, load_legacy_images

# Load all images for a repository
images = load_repository_images("pytorch-training")  # list[ImageConfig]

# Load legacy support policy images
legacy = load_legacy_images()  # dict[str, list[ImageConfig]]
```

### Sorting Functions

```python
from image_config import sort_by_version

# Sort images by version descending
sorted_images = sort_by_version(images)

# Sort with tiebreakers (for available_images: platform, then accelerator)
sorted_images = sort_by_version(
    images,
    tiebreakers=[
        lambda img: 0 if img.get("platform") == "sagemaker" else 1,
        lambda img: {"gpu": 0, "neuronx": 1, "cpu": 2}.get(img.get("accelerator", "").lower(), 3),
    ],
)
```

### Table Building

```python
from image_config import build_image_row
from utils import render_table

# Build a complete row (uses img.get_display() internally)
row = build_image_row(img, columns)  # ["PyTorch 2.9", "py312", ...]

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
   # Image tags (first tag is used in available_images.md)
   tags:
     - "2.9.0-gpu-py312-cu130-ubuntu22.04-ec2"
   ```

1. Optional fields:

   ```yaml
   cuda: cu130                  # For GPU images
   sdk: "2.27.1"                # For Neuron images
   os: ubuntu22.04              # Operating system
   public_registry: true        # If in ECR Public Gallery
   ga: "2025-10-15"             # GA date (for support policy)
   eop: "2026-10-15"            # EOP date (for support policy)
   example_ecr_account: "007439368137"  # Override default ECR account
   ```

1. GA/EOP dates are only needed for repositories that appear in support policy (base, pytorch-*, tensorflow-* excluding neuronx).

### Adding Release Notes to an Image

To generate release notes for an image, add the following fields to the image config YAML:

```yaml
# Required for release notes generation (both must be present)
announcement:
  - "Introduced containers for PyTorch 2.9 for Training"
  - "Added Python 3.12 support"

packages:
  python: "3.12"
  pytorch: "2.9.0"
  cuda: "13.0"
  cudnn: "9.13.0.50"
  nccl: "2.27.7-1"

# Optional sections (rendered dynamically at end of release notes)
optional:
  known_issues:
    - "Description of known issue"
```

**Required fields for release notes** are defined in `constants.py`:

```python
RELEASE_NOTES_REQUIRED_FIELDS = ["announcement", "packages"]
```

The `packages` field uses keys that map to display names in `global.yml` under `display_names`.

### Release Notes Generation

Release notes are automatically generated for images that have all required fields (`announcement` and `packages`).

**Output structure:**

```
docs/releasenotes/
├── index.md                    # Main release notes index
├── <framework_group>/          # e.g., pytorch/, tensorflow/, base/
│   ├── index.md                # Framework-specific index with table of links
│   └── <repo>-<ver>-<acc>-<plat>.md  # Individual release notes
└── archive/                    # Historical manual release notes
    ├── index.md
    ├── pytorch/
    ├── tensorflow/
    └── ...
```

**Framework index page structure:**

The framework-specific `index.md` uses table rendering (like `available_images.md` and `support_policy.md`) with columns defined in `tables/extra/release_notes.yml`:

- Repository (displayed as Framework), Accelerator, Platform, Link

The index separates release notes into two sections:

1. **Supported images** - Images where `eop >= today` or no EOP date is set
1. **Deprecated images** - Images where `eop < today`, displayed in a warning admonition

This separation uses the `ImageConfig.is_supported` property.

**Generated release note sections:**

1. **Announcement** - Bullet list from `announcement` field
1. **Core Packages** - Table from `packages` field (keys mapped via `display_names`)
1. **Security Advisory** - Hardcoded section with link to AWS Security Bulletin
1. **Reference** - Docker image URIs (private ECR + public ECR if `public_registry: true`) and links to available_images.md and support_policy.md
1. **Known Issues** (optional) - Bullet list from `known_issues` field

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
dlc_long: "${aws} ${dlc}"            # -> "AWS Deep Learning Containers"
sagemaker: "${amazon} SageMaker AI"  # -> "Amazon SageMaker AI"

# Platform display mappings
platforms:
  ec2: "EC2, ECS, EKS"
  sagemaker: "SageMaker"

# Consolidated display names (repositories, frameworks, packages, optional sections)
display_names:
  # Framework groups
  pytorch: "PyTorch"
  tensorflow: "TensorFlow"
  # Repositories
  pytorch-training: "PyTorch Training"
  # Packages (for release notes Core Packages table)
  python: "Python"
  cuda: "CUDA"
  cudnn: "cuDNN"
  # Optional sections
  known_issues: "Known Issues"

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

# Generate specific outputs
python main.py --available-images-only
python main.py --support-policy-only
python main.py --release-notes-only

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
- The `ImageConfig.is_supported` property checks if `eop >= today`

## Update knowledge base

If there are any new changes to the documentations generation and organization, make sure to update you knowledge base in the steering/docs.md file and any runbook or update to processes should also be updated in DEVELOPMENT.md files.
This is done so that developers get the most up-to-date information on the current codebase. Be sure to not let this steering document get too large since it will overflow the context window.
If the document gets longer than 500 lines, make sure to delete unnecessary sections and condense verbose sections where necessary. Also, do not delete comments unless the it is unnecessary.

When making changes to image config field reading or processing, update `docs/src/data/template/image-template.yml` to reflect those changes.
