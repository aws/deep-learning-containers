# Lambda GPU Image Versions

## Versioning Policy

Each image carries a `LABEL dlc_major_version` in the Dockerfile. Minor versions are auto-incremented during release.

A **major version bump** is required when any core component changes:

- Python version (e.g. 3.13 → 3.14)
- CUDA version (e.g. 12.8 → 12.9)
- Main framework version (e.g. PyTorch minor/major bump, CuPy major bump)
- Other breaking changes (base OS upgrade, Lambda RIC API changes, etc.)

Dependency-only updates (security patches, minor pip package bumps) increment the minor version automatically and do not require a major version bump.

## Core Components (v1)

| Component    | Version                                           | All Images |
| ------------ | ------------------------------------------------- | :--------: |
| CUDA         | 12.8.1 (runtime)                                  |     ✓      |
| Python       | 3.13                                              |     ✓      |
| Amazon Linux | 2023                                              |     ✓      |
| Lambda RIC   | latest (from `public.ecr.aws/lambda/python:3.13`) |     ✓      |

## Image-Specific Components

### gpu-base-py3

Minimal CUDA + Python + Lambda runtime. No ML libraries.

### gpu-cupy-py3

| Package | Version |
| ------- | ------- |
| CuPy    | 14.0.1  |
| NumPy   | 2.4.2   |
| SciPy   | 1.17.1  |
| Pandas  | 3.0.1   |
| Numba   | 0.64.0  |
| cvxpy   | 1.8.1   |

### gpu-pytorch-py3

| Package      | Version              |
| ------------ | -------------------- |
| PyTorch      | 2.10.0               |
| TorchVision  | 0.25.0               |
| TorchAudio   | 2.10.0               |
| Transformers | 5.2.0                |
| Diffusers    | 0.36.0               |
| SAM2         | 2b90b9f (git)        |
| Accelerate   | 1.12.0               |
| OpenCV       | 4.13.0.92            |
| FFmpeg       | n8.0.1 (NVENC/NVDEC) |
| NumPy        | 2.4.2                |
| SciPy        | 1.17.1               |
