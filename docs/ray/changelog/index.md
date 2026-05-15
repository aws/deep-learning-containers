# Changelog

Changelog for the Ray Serve DLC images.

* * *

## v1.1.0 — 2026-05-12

**Tags:** `serve-ml-cuda-v1.1` · `serve-ml-cpu-v1.1` · `serve-ml-sagemaker-cuda-v1.1` · `serve-ml-sagemaker-cpu-v1.1`

### Changes

- Upgraded Ray from 2.54.1 to 2.55.1
- Upgraded Transformers from 5.5.4 to 5.8.0

* * *

## v1.0.0 — 2026-04-23

**Tags:** `serve-ml-cuda-v1.0` · `serve-ml-cpu-v1.0` · `serve-ml-sagemaker-cuda-v1.0` · `serve-ml-sagemaker-cpu-v1.0`

### Highlights

- Initial release of Ray Serve DLC images on Amazon Linux 2023
- GPU and CPU variants for both EC2 and SageMaker
- Ray 2.54.1 with PyTorch, CUDA 12.9, Python 3.13
- Model package structure: `config.yaml` + `deployment.py` + optional `code/requirements.txt`
- SageMaker adapter with `/ping` and `/invocations` endpoints
- Automatic runtime dependency installation from `requirements.txt`
- CodeArtifact support for private dependencies
- Security-hardened Python build
