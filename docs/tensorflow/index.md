# ML Training using TensorFlow DLC

Production-ready Docker images for TensorFlow training workloads on {{ aws }}. Available in CPU and GPU variants, built on Amazon Linux 2023 with
ongoing security patching.

These images bundle TensorFlow with the libraries needed for **distributed training on {{ sagemaker }}** — EFA for low-latency networking on
EFA-capable instances, OpenMPI for multi-node coordination, and the SageMaker training toolkits for entry-point invocation and channel wiring.

## Images

| Platform        | Variant | Image                                                                                                 |
| --------------- | ------- | ----------------------------------------------------------------------------------------------------- |
| {{ sagemaker }} | GPU     | `public.ecr.aws/deep-learning-containers/tensorflow-training:2.21-gpu-py312-cu129-amzn2023-sagemaker` |
| {{ sagemaker }} | CPU     | `public.ecr.aws/deep-learning-containers/tensorflow-training:2.21-cpu-py312-amzn2023-sagemaker`       |

All images are also available on the [ECR Public Gallery](https://gallery.ecr.aws/deep-learning-containers/tensorflow-training). For private ECR URIs,
see [Image Access](../get_started/index.md).

The `2.21.0-*` fully-qualified tags (e.g. `2.21.0-gpu-py312-cu129-amzn2023-sagemaker`) are also published and pin to the same image digest.

## What's Included

The GPU image includes:

- **TensorFlow 2.21.0** (`tensorflow==2.21.0`)
- **CUDA 12.9.1** with **cuDNN 9.5.1.17** (`nvidia-cudnn-cu12`) and **NCCL 2.30.7** (`nvidia-nccl-cu12`) for multi-GPU collectives
- **[EFA](https://aws.amazon.com/hpc/efa/) 1.49.0** with **OpenMPI 4.1.8** for low-latency multi-node communication on EFA-capable instances
- **[MPI for Python](https://mpi4py.readthedocs.io/) (`mpi4py`)** for multi-process Python coordination
- **[SageMaker training toolkits](https://github.com/aws/sagemaker-tensorflow-training-toolkit)** — `sagemaker-tensorflow-training`, `sagemaker-training`
- **[MLflow](https://mlflow.org/) 3.9+**, **`sagemaker>=3.4.0`**, **`smclarify`**, and **`sagemaker-experiments` 0.1.45**
- **Data & ML tooling** — `tensorflow-io` 0.37, `tensorflow-datasets`, `pandas`, `scikit-learn`, `scipy`, `numpy` (2.1+), `Pillow`, `h5py`,
  `opencv-python`, `numba`, `plotly`, `seaborn`, `shap`, `bokeh`, `imageio`, `cloudpickle`
- **AWS tooling** — `boto3`, `botocore`, `awscli` (<2), `s3fs`
- **SageMaker Studio integration** — `sagemaker-studio-analytics-extension`, `sparkmagic` 0.22.0, `sagemaker-studio-sparkmagic-lib`
- **Python 3.12** in a venv at `/opt/venv` (`PATH` already set)

The CPU variant includes the same TensorFlow ecosystem and SageMaker toolkits. EFA, CUDA, cuDNN, and NCCL are not present in the CPU image.

## CUDA Forward Compatibility

The GPU image entrypoint detects host NVIDIA driver versions older than the bundled `cuda-compat` layer and automatically prepends
`/usr/local/cuda/compat` to `LD_LIBRARY_PATH`. No flag or env var needed — the check runs on every container start.

## How We Build

These images are curated builds tracking the [TensorFlow](https://www.tensorflow.org/) project:

- **Built from upstream TensorFlow wheels** published to PyPI
- **Reproducible** — pinned via `pyproject.toml` + `uv.lock` for every image variant
- **Security-patched** — continuously maintained with security patches from {{ aws }} on an Amazon Linux 2023 base

TensorFlow 2.21 is the first TensorFlow DLC on Amazon Linux 2023 and Python 3.12. Prior TensorFlow DLCs shipped on Ubuntu with earlier Python versions;
this release moves the base OS and interpreter forward alongside the framework bump.
