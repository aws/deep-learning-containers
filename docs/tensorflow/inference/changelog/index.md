# Changelog

Changelog for the Amazon Linux 2023-based TensorFlow Serving SageMaker inference images (`2.20-gpu-py312-cu129-amzn2023-sagemaker` and
`2.20-cpu-py312-amzn2023-sagemaker`).

* * *

## TensorFlow Serving 2.20 — 2026-08-20

**Tags:** `2.20-gpu-py312-cu129-amzn2023-sagemaker` · `2.20-cpu-py312-amzn2023-sagemaker` · `2.20.0-gpu-py312-cu129-amzn2023-sagemaker` ·
`2.20.0-cpu-py312-amzn2023-sagemaker`

**Bundled versions:** TensorFlow Serving 2.20.0 · Python 3.12 · CUDA 12.9.1 · cuDNN 9.24.0.43 · nginx 1.30.3 · njs 0.9.9

### Highlights

- Initial release of the TensorFlow inference DLC on Amazon Linux 2023 and Python 3.12
- TensorFlow Serving 2.20.0 for SageMaker AI inference (CPU and GPU variants)
- CUDA 12.9.1 with cuDNN 9.24.0.43 on the GPU variant, with automatic CUDA forward compatibility on older host drivers
- nginx 1.30.3 with the njs 0.9.9 module fronting the model server on port 8080
- Multi-model endpoints supported (`SAGEMAKER_MULTI_MODEL=true`), including per-model `inference.py` handlers
- Server-side batching, thread-pool sizing, and multiple model server processes configurable via `SAGEMAKER_TFS_*` environment variables
- The `tensorflow` framework wheel, EFA, OpenMPI, and NCCL are intentionally excluded — only the TensorFlow Serving gRPC stubs ship
  (`tensorflow-serving-api-gpu` on GPU, `tensorflow-serving-api` on CPU), keeping the image small
