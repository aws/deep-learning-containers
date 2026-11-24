# Changelog

Changelog for the Amazon Linux 2023-based TensorFlow SageMaker training images (`2.21-gpu-py312-cu129-amzn2023-sagemaker` and
`2.21-cpu-py312-amzn2023-sagemaker`).

* * *

## TensorFlow 2.21 — Blackwell support — 2026-11-21

**Tags:** `2.21-gpu-py312-cu129-amzn2023-sagemaker` · `2.21.0-gpu-py312-cu129-amzn2023-sagemaker`

### Highlights

- Native NVIDIA Blackwell support on the GPU variant: the TensorFlow 2.21.0 wheel is now built from source for GPU targets `sm_60`, `sm_70`, `sm_80`,
  `sm_89`, `sm_90`, `sm_100` and `sm_120`, plus `compute_120` PTX, so training runs on G7 and G7e instances without PTX JIT compilation
- Validated with SageMaker AI training jobs on `ml.g7.2xlarge` and `ml.g7.12xlarge` (`MirroredStrategy` across 2 GPUs)

* * *

## TensorFlow 2.21 — 2026-07-10

**Tags:** `2.21-gpu-py312-cu129-amzn2023-sagemaker` · `2.21-cpu-py312-amzn2023-sagemaker` · `2.21.0-gpu-py312-cu129-amzn2023-sagemaker` ·
`2.21.0-cpu-py312-amzn2023-sagemaker`

**Bundled versions:** TensorFlow 2.21.0 · Python 3.12 · CUDA 12.9.1 · cuDNN 9.24.0.43 · NCCL 2.30.7 · EFA 1.49.0 · OpenMPI 4.1.8

### Highlights

- Initial release of the TensorFlow DLC on Amazon Linux 2023 and Python 3.12
- TensorFlow 2.21.0 for SageMaker AI training (CPU and GPU variants)
- CUDA 12.9.1 with cuDNN 9.24.0.43 and NCCL 2.30.7 on the GPU variant
- EFA 1.49.0 with OpenMPI 4.1.8 for multi-node training on EFA-capable instances
- SageMaker toolkits pre-installed: `sagemaker-tensorflow-training`, `sagemaker-training`
- MLflow 3.9+, `smclarify`, `sagemaker>=3.4.0`, and SageMaker Studio integrations bundled
