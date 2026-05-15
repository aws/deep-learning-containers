# PyTorch Training

Production-ready Docker images for PyTorch training workloads on {{ aws }}. Available in CPU and GPU variants, built on Amazon Linux 2023 with ongoing
security patching.

These images bundle PyTorch with the libraries needed for **distributed training at scale** — EFA for low-latency networking, NCCL for multi-GPU
collectives, flash-attn and Transformer Engine for fused attention/FP8 kernels, and DeepSpeed for memory-efficient large-model training.

## Images

| Platform | Variant | Image |
| --- | --- | --- |
| EC2 / EKS | GPU | `public.ecr.aws/deep-learning-containers/pytorch:2.11-cu130-amzn2023` |
| EC2 / EKS | CPU | `public.ecr.aws/deep-learning-containers/pytorch:2.11-cpu-amzn2023` |
| Amazon SageMaker AI | GPU | `public.ecr.aws/deep-learning-containers/pytorch:2.11-cu130-amzn2023-sagemaker` |
| Amazon SageMaker AI | CPU | `public.ecr.aws/deep-learning-containers/pytorch:2.11-cpu-amzn2023-sagemaker` |

For private ECR URIs, see [Image Access](../get_started/index.md).

## What's Included

The GPU images bundle the full distributed-training stack so you can launch multi-GPU and multi-node training without building a custom image:

- **PyTorch 2.11.0** with `torchvision` 0.26.0 and `torchaudio` 2.11.0 (CUDA 13.0 wheels for the GPU variant, CPU wheels for the CPU variant)
- **CUDA 13.0.2** with cuDNN and **NCCL 2.26.2** for multi-GPU collectives
- **[EFA](https://aws.amazon.com/hpc/efa/) 1.47.0** with **OpenMPI** and the **AWS NCCL OFI plugin** for low-latency multi-node communication on
  EFA-capable instances
- **[GDRCopy](https://github.com/NVIDIA/gdrcopy) 2.4.4** userspace library for direct GPU-to-NIC memory copies
- **[flash-attn](https://github.com/Dao-AILab/flash-attention) 2.8.3** — fused attention kernels for transformer training
- **[Transformer Engine](https://github.com/NVIDIA/TransformerEngine) 2.12.0** — FP8/BF16 mixed-precision primitives optimized for Hopper and newer
  GPUs
- **[DeepSpeed](https://www.deepspeed.ai/) 0.18.8** — ZeRO sharding, pipeline parallel, and memory-efficient optimizers
- **[FastAI](https://www.fast.ai/)**, `boto3`, `botocore`, `requests`, `PyYAML`, `GitPython`, `Mako`
- **NCCL test utility** — `all_reduce_perf` is pre-installed at `/usr/local/bin/all_reduce_perf` for verifying EFA/NCCL connectivity before training
- **OpenSSH** server pre-configured (port 22) for inter-node communication in MPI/`torchrun` launches
- **Python 3.12** in a venv at `/opt/venv` (`PATH` already set)

The CPU variant includes the same PyTorch ecosystem plus `mpi4py`, `scipy`, `scikit-learn`, and `opencv-python`. EFA, flash-attn, Transformer Engine,
and GDRCopy are not present in the CPU image.

The SageMaker variants additionally bundle `sagemaker`, `sagemaker-pytorch-training`, `sagemaker-training`, `mlflow`, `smclarify`, `s3fs`, `shap`,
`pandas`, `seaborn`, and other SageMaker-specific dependencies.

## CUDA Forward Compatibility

The GPU image entrypoint detects host NVIDIA driver versions older than the bundled `cuda-compat` layer and automatically prepends
`/usr/local/cuda/compat` to `LD_LIBRARY_PATH`. No flag or env var needed — the check runs on every container start.

## How We Build

These images are curated builds tracking the [PyTorch](https://pytorch.org/) project:

- **Built from upstream PyTorch wheels** with our own compiled flash-attn / Transformer Engine layered on top
- **Reproducible** — pinned via `pyproject.toml` + `uv.lock`, with build wheels cached in S3 across CI runs
- **Security-patched** — continuously maintained with security patches from {{ aws }} on an Amazon Linux 2023 base
