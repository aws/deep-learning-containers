# Changelog

Changelog for the Amazon Linux 2023-based Base images (`runtime-cuda`, `devel-cuda`).

* * *

## v2.0.0 — 2026-04-30

**Tags:** `runtime-cuda-v2` · `devel-cuda-v2`

### Highlights

- Initial release of the v2 (CUDA 13.0) Base image line on Amazon Linux 2023
- CUDA 13.0.2 with Python 3.13.12 (built from source with hardening flags)
- `uv` package manager pre-installed for fast dependency resolution
- Both `runtime` (minimal) and `devel` (with `nvcc`, headers, build tools) variants

* * *

## v1.0.0 — 2026-04-30

**Tags:** `runtime-cuda-v1` · `devel-cuda-v1`

### Highlights

- Initial release of the lightweight Base image line on Amazon Linux 2023
- CUDA 12.9.1 with Python 3.13.12 (built from source with hardening flags)
- Both `runtime` (minimal CUDA libraries) and `devel` (full CUDA Toolkit + `gcc`, `cmake`, `git`, `make`) variants
- `uv` package manager pre-installed at `/usr/local/bin/uv`
- Continuous security patching via `dnf upgrade --security`
