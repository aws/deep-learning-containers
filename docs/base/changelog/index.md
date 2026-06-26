# Changelog

Changelog for the Amazon Linux 2023-based Base images (`runtime`, `devel`).

* * *

## CUDA 13.0 — 2026-04-30

**Tags:** `runtime-cu130-amzn2023` · `devel-cu130-amzn2023`

### Highlights

- Initial release of the CUDA 13.0 Base image line on Amazon Linux 2023
- CUDA 13.0.2 with Python 3.13.12 (built from source with hardening flags)
- `uv` package manager pre-installed for fast dependency resolution
- Both `runtime` (minimal) and `devel` (with `nvcc`, headers, build tools) variants

* * *

## CUDA 12.9 — 2026-04-30

**Tags:** `runtime-cu129-amzn2023` · `devel-cu129-amzn2023`

### Highlights

- Initial release of the lightweight Base image line on Amazon Linux 2023
- CUDA 12.9.1 with Python 3.13.12 (built from source with hardening flags)
- Both `runtime` (minimal CUDA libraries) and `devel` (full CUDA Toolkit + `gcc`, `cmake`, `git`, `make`) variants
- `uv` package manager pre-installed at `/usr/local/bin/uv`
- Continuous security patching via `dnf upgrade --security`
