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
├── generate.py                    # Generation logic
├── global.yml                     # Shared terminology and configuration
├── hooks.py                       # MkDocs hooks
├── logger.py                      # Logging configuration
├── macros.py                      # MkDocs macros
├── main.py                        # CLI entry point
└── utils.py                       # Reusable helper functions
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
# Development
cd docs && source .venv/bin/activate
cd src && python main.py --verbose

# With MkDocs (automatic via hooks)
mkdocs serve
mkdocs build
```

### Support Policy Logic

- Scans all image configs for `ga` and `eop` fields
- Groups by (repository, version)
- Validates consistency within each group
- Auto-determines supported/unsupported by comparing `eop` to current date
- Uses `display_names` for Framework column

### Legacy Support Data

Historical support policy data for older, unsupported images is stored in `docs/src/legacy/legacy_support.yml`.

#### Directory Structure

```
docs/src/legacy/
└── legacy_support.yml    # Historical support data
```

#### File Format

```yaml
PyTorch:
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
- The `is_image_supported()` function checks if `eop >= today`

## Code Organization

### File Responsibilities

- `generate.py` - Contains only `generate_*` functions for documentation generation
- `utils.py` - Contains all reusable helper functions
- `macros.py` - MkDocs macros plugin integration
- `hooks.py` - MkDocs hooks entry point
- `constants.py` - All path constants and global variables

### utils.py Conventions

All functions in `utils.py` must:

1. Have a docstring explaining what the function does
1. Be fully typed with type hints
1. Document exceptions raised (if any)

Example:

```python
def get_display_name(global_config: dict, repository: str) -> str:
    """Get human-readable display name for a repository.

    Raises:
        KeyError: If repository not found in global config display_names.
    """
```

## Update knowledge base

If there are any new changes to the documentations generation and organization, make sure to update you knowledge base in the steering/docs.md file.
