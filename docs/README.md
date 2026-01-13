# Documentation Development Runbook

## Prerequisites

Follow the environment setup instructions in [DEVELOPMENT.md](../DEVELOPMENT.md), then install documentation dependencies:

```bash
pip install -r docs/requirements.txt
```

## Local Development

Start the development server with live reload:

```bash
mkdocs serve --livereload
```

## Tutorials Setup (Local Only)

Clone the tutorials repository into `docs/tutorials/` for local development:

```bash
git clone https://github.com/aws-samples/sample-aws-deep-learning-containers.git docs/tutorials
```

This step is only required for local development. GitHub Actions automatically clones this repository during production deployment.
For any changes required to the tutorial pages, create a new PR in [aws-samples/sample-aws-deep-learning-containers](https://github.com/aws-samples/sample-aws-deep-learning-containers.git).

## Navigation

Site navigation is managed centrally in `docs/.nav.yml` using the `awesome-nav` plugin. Structure:

```yaml
nav:
  - Home: index.md
  - Section Name:
      - section/index.md
      - Page Title: section/page.md
  - Directory Reference: dirname  # Auto-discovers pages in directory
```

## Configuration

Key settings in `mkdocs.yaml`:

**Theme Palette** - Modify color scheme under `theme.palette`:

```yaml
theme:
  palette:
    - scheme: default    # light mode
      primary: custom
      accent: custom
```

**Plugins** - Add/remove plugins under `plugins`:

```yaml
plugins:
  - search
  - autorefs
  - awesome-nav
```
