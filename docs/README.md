# Documentation Website Guide

Guide for setting up, running, and configuring the MkDocs documentation site. For adding or modifying generated content (images, release notes, support policy), see [DEVELOPMENT.md](DEVELOPMENT.md).

## Prerequisites

```bash
# Set up virtual environment from repository root
cd /path/to/deep-learning-containers
python -m venv .venv
source .venv/bin/activate
pip install -r docs/requirements.txt
```

## Local Development

### Generation Only

Run the generation system without serving:
```bash
cd docs/src && python main.py --verbose
```
This clones the `tutorials/` repository and generates markdown files in `reference/` and `releasenotes/` directories.

Generation flags:
```bash
python main.py --available-images-only
python main.py --support-policy-only
python main.py --release-notes-only
python main.py --index-only
```

### Serving

Use `mkdocs serve` to generate documentation and serve the website:
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

