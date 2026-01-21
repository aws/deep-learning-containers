# Documentation Development Guide

This guide provides runbooks for developers working with the {{ dlc_long }} documentation.

## Directory Structure

```
docs/
├── src/
│   ├── data/                    # Per-image configuration files
│   │   ├── pytorch-training/
│   │   │   ├── 2.9-gpu-ec2.yml
│   │   │   └── 2.9-cpu-sagemaker.yml
│   │   └── <repository>/
│   │       └── <version>-<accelerator>-<platform>.yml
│   ├── tables/                  # Table column configurations
│   │   └── <repository>.yml
│   ├── templates/reference/     # Jinja2 templates
│   │   ├── available_images.template.md
│   │   └── support_policy.template.md
│   ├── global.yml               # Shared terminology and configuration
│   ├── generate.py              # Generation logic
│   └── main.py                  # CLI entry point
├── reference/
│   ├── available_images.md      # Generated
│   └── support_policy.md        # Generated
└── .venv/                       # Python virtual environment
```

---

## Adding a New Image to available_images.md

### Step 1: Create Image Configuration File

Create a YAML file in `docs/src/data/<repository>/`:

**File naming convention:**

- Standard images: `<version>-<accelerator>-<platform>.yml`
- Neuron images: `<version>-neuronx-sdk<sdk_version>.yml`

**Examples:**

- `2.9-gpu-ec2.yml`
- `2.9-cpu-sagemaker.yml`
- `2.7-neuronx-sdk2.24.1.yml`

### Step 2: Add Required Fields

```yaml
# Required fields
framework: PyTorch           # Display name for the framework
version: '2.9'               # Framework version (quote to preserve as string)
accelerator: gpu             # gpu, cpu, or neuronx
python: py312                # Python version (e.g., py310, py312)
platform: ec2                # ec2 or sagemaker
tag: 2.9.0-gpu-py312-cu130-ubuntu22.04-ec2    # Full image tag
```

### Step 3: Add Optional Fields

```yaml
# Optional fields
cuda: cu130                  # CUDA version (for GPU images)
sdk: 2.24.1                  # Neuron SDK version (for neuronx images)
os: ubuntu22.04              # Operating system
public_registry: true        # Set to true if available in ECR Public Gallery
example_ecr_account: '007439368137'  # Override default ECR account (763104351884)
ga: '2025-10-15'             # GA date (for support policy)
eop: '2026-10-15'            # End of Patch date (for support policy)
```

### Step 4: Regenerate Documentation

```bash
cd docs && source .venv/bin/activate
cd src && python main.py --verbose
```

---

## Adding Support Policy for an Image

Support policy entries are automatically generated from `ga` and `eop` fields in image configs.

### Which Images Need GA/EOP Dates?

Only images in these repositories require GA/EOP dates:

- `base`
- `pytorch-training`, `pytorch-training-arm64`
- `pytorch-inference`, `pytorch-inference-arm64`
- `tensorflow-training`
- `tensorflow-inference`, `tensorflow-inference-arm64`

**Note:** Neuron images (`*-neuronx`) do NOT require GA/EOP dates.

### Adding GA/EOP Dates

Add to your image config file:

```yaml
ga: '2025-10-15'    # General Availability date (YYYY-MM-DD)
eop: '2026-10-15'   # End of Patch date (YYYY-MM-DD)
```

### Validation Rules

- All images with the same (repository, version) must have identical GA/EOP dates
- The generator validates this and raises an error if dates are inconsistent
- Supported vs Unsupported is auto-determined by comparing EOP to current date

---

## Editing Table Columns in available_images.md

Table columns are configured in `docs/src/tables/<repository>.yml`.

### Modify Column Order

Edit the `columns` list order:

```yaml
# docs/src/tables/pytorch-training.yml
columns:
  - field: framework_version    # First column
    header: Framework
  - field: python               # Second column
    header: Python
  - field: cuda                 # Third column
    header: CUDA
  # ... remaining columns
```

### Add a New Column

Add a new entry to the `columns` list:

```yaml
columns:
  - field: framework_version
    header: Framework
  - field: os                   # New column
    header: Operating System
  - field: python
    header: Python
```

### Available Fields

| Field               | Description                                              |
| ------------------- | -------------------------------------------------------- |
| `framework_version` | Framework name + version (e.g., "PyTorch 2.9")           |
| `python`            | Python version                                           |
| `cuda`              | CUDA version                                             |
| `sdk`               | Neuron SDK version                                       |
| `accelerator`       | gpu, cpu, or neuronx                                     |
| `platform`          | Displayed platform (from `global.yml` platforms mapping) |
| `os`                | Operating system                                         |
| `example_url`       | ECR image URL                                            |

### Remove a Column

Simply remove the entry from the `columns` list.

---

## Reordering Tables in available_images.md

Tables appear in the order specified in `docs/src/global.yml` under `table_order`.

### Change Table Order

Edit the `table_order` list:

```yaml
# docs/src/global.yml
table_order:
  - base                    # First table
  - vllm                    # Second table
  - pytorch-training        # Third table
  # ... remaining tables
```

### Hide a Table

Remove the repository from `table_order`. The images will still exist but won't appear in generated docs.

---

## Adding a New Repository

### Step 1: Create Data Directory

```bash
mkdir -p docs/src/data/<repository-name>
```

### Step 2: Create Table Configuration

Create `docs/src/tables/<repository-name>.yml`:

```yaml
columns:
  - field: framework_version
    header: Framework
  - field: python
    header: Python
  - field: accelerator
    header: Accelerator
  - field: platform
    header: Platform
  - field: example_url
    header: Example URL
```

### Step 3: Add Display Name (Required)

Add to `docs/src/global.yml` under `display_names`:

```yaml
display_names:
  # ... existing entries
  my-new-repo: My New Repository
```

**Note:** An error is raised if a repository is missing from `display_names`.

### Step 4: Add to Table Order

Add to `docs/src/global.yml` under `table_order`:

```yaml
table_order:
  # ... existing entries
  - my-new-repo
```

### Step 5: Add Image Configs

Create image config files in `docs/src/data/<repository-name>/`.

---

## Using Global Variables in Markdown

The documentation uses mkdocs-macros-plugin for variable substitution.

### Available Variables

Variables are defined in `docs/src/global.yml` and exposed via `docs/src/macros.py`:

| Variable          | Value                        |
| ----------------- | ---------------------------- |
| `{{ aws }}`       | AWS                          |
| `{{ amazon }}`    | Amazon                       |
| `{{ dlc }}`       | Deep Learning Containers     |
| `{{ dlc_long }}`  | AWS Deep Learning Containers |
| `{{ dlc_short }}` | DLC                          |
| `{{ sagemaker }}` | Amazon SageMaker AI          |
| `{{ ec2 }}`       | Amazon EC2                   |
| `{{ ecs }}`       | Amazon ECS                   |
| `{{ eks }}`       | Amazon EKS                   |
| `{{ ecr }}`       | Amazon ECR                   |

### Usage in Markdown

```markdown
# Welcome to {{ dlc_long }}

Deploy on {{ sagemaker }}, {{ eks }}, or {{ ec2 }}.
```

### Adding New Variables

1. Add to `docs/src/global.yml`:

    ```yaml
    my_var: My Value
    ```

1. Variables are automatically exposed if they are strings.

---

## Image Sorting in Tables

Images are automatically sorted by:

1. **Version** (descending) - Newest versions first
1. **Platform** - SageMaker before EC2
1. **Accelerator** - GPU before NeuronX before CPU

This sorting is handled in `docs/src/generate.py` and cannot be configured.

---

## Running Documentation Generation

### Prerequisites

```bash
cd docs
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Generate Documentation

```bash
cd docs/src
python main.py --verbose
```

### Preview with MkDocs

```bash
cd /path/to/deep-learning-containers
mkdocs serve
```

Open http://127.0.0.1:8000 in your browser.

### Build Static Site

```bash
mkdocs build
```

---

## Overriding ECR Account

Some images use a different ECR account than the default (`763104351884`).

### Per-Image Override

Add `example_ecr_account` to the image config:

```yaml
# docs/src/data/sagemaker-tritonserver/25.04-gpu-sagemaker.yml
framework: Triton
version: '25.04'
accelerator: gpu
platform: sagemaker
tag: 25.04-py3
example_ecr_account: '007439368137'  # Override default account
```

---

## Troubleshooting

### "Repository not found in display_names"

Add the repository to `display_names` in `docs/src/global.yml`.

### "Inconsistent GA/EOP dates"

All images with the same (repository, version) must have identical GA/EOP dates. Check your image configs.

### "Variable not found"

Ensure the variable is defined in `global.yml` and is a string type. Non-string values (dicts, lists) are not exposed to macros.

### Images Not Appearing

1. Check the image config file exists in `docs/src/data/<repository>/`
1. Verify the repository is in `table_order` in `global.yml`
1. Run `python main.py --verbose` to see any errors
