# Distributed Training using Ray Train DLC

Production-ready Docker image for distributed training with [Ray Train](https://docs.ray.io/en/latest/train/train.html) on {{ aws }}. Built on Amazon
Linux 2023 with ongoing security patching.

This is the PyTorch training stack plus Ray. It carries the same interconnect and kernel layer as the [PyTorch DLC](../pytorch/index.md) — EFA for
low-latency networking, NCCL for multi-GPU collectives, flash-attn and Transformer Engine for fused attention/FP8, DeepSpeed for memory-efficient
large-model training — with Ray Train, Tune, and Data layered on top. One image runs as either a Ray head or a Ray worker, so the same tag serves
KubeRay clusters on {{ eks }} and manually bootstrapped clusters on {{ ec2_short }}.

## Images

| Platform | Variant | Image |
| --- | --- | --- |
| {{ ec2_short }} / {{ eks_short }} | GPU | `public.ecr.aws/deep-learning-containers/ray:train-ml-cuda` |

Ray Train shares the `ray` repository with the [Ray Serve DLC](../ray/index.md), using the `train-ml` tag prefix. Versioned tags (e.g.
`train-ml-cuda-v1`, `train-ml-cuda-v1.0`, and `train-ml-cuda-v1.0.0`) are published alongside the floating tag. The image is also available on the
[ECR Public Gallery](https://gallery.ecr.aws/deep-learning-containers/ray). For private ECR URIs, see [Image Access](../get_started/index.md).

## What's Included

The image bundles the full distributed-training stack so you can launch multi-GPU and multi-node Ray Train jobs without building a custom image:

- **[Ray](https://docs.ray.io/) 2.58.0** with the `default`, `train`, `tune`, and `data` extras — `ray[default]` supplies the dashboard and the
  job-submission server that `ray job submit` and KubeRay's health probes use
- **[PyTorch](https://pytorch.org/) 2.13.0** with `torchvision` 0.28.0 (CUDA 13.0 wheels)
- **CUDA 13.0.2** with cuDNN and **NCCL 2.29.7** for multi-GPU collectives
- **[EFA](https://aws.amazon.com/hpc/efa/) 1.47.0** with **OpenMPI** and the **AWS NCCL OFI plugin** for low-latency multi-node communication on
  EFA-capable instances
- **[GDRCopy](https://github.com/NVIDIA/gdrcopy) 2.4.4** userspace library for direct GPU-to-NIC memory copies
- **[flash-attn](https://github.com/Dao-AILab/flash-attention) 2.8.3** — fused attention kernels for transformer training
- **[Transformer Engine](https://github.com/NVIDIA/TransformerEngine) 2.13.0** — FP8/BF16 mixed-precision primitives
- **[DeepSpeed](https://www.deepspeed.ai/) 0.19.2** — ZeRO sharding, pipeline parallel, and memory-efficient optimizers
- **[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) 2.6.5** — backs Ray Train's `RayFSDPStrategy` and `RayDDPStrategy` integrations
- **[Transformers](https://huggingface.co/docs/transformers) 5.13.0**, `datasets` 5.0.0, `accelerate` 1.14.0, `evaluate` 0.4.6, `torchmetrics` 1.9.0 —
  the Hugging Face training ecosystem
- **scikit-learn, NumPy, pandas**, and `boto3` / `awscli`
- **NCCL test utility** — `all_reduce_perf` is pre-installed at `/usr/local/bin/all_reduce_perf` for verifying EFA/NCCL connectivity before training
- **OpenSSH** server pre-configured (port 22) for inter-node communication in MPI launches
- **Python 3.13** in a venv at `/opt/venv` (`PATH` already set)

For model serving with Ray Serve, use the [Ray Serve DLC](../ray/index.md).

## Cluster Roles and Ports

The image does not decide whether it is a head or a worker, so the same tag works under any orchestrator. Under KubeRay the `RayCluster` spec supplies
the `ray start` command for each pod; on {{ ec2_short }} you run `ray start` yourself. The head listens on **6379** (GCS, where workers connect),
**8265** (dashboard and job-submission API), and **10001** (Ray Client).

`FI_PROVIDER=efa` and `NCCL_DEBUG=INFO` are set in the image, and `/etc/nccl.conf` carries `NCCL_SOCKET_IFNAME=^docker0,lo` so NCCL auto-detects the
right interface on any host. Override `NCCL_SOCKET_IFNAME` per platform when you need to — {{ eks_short }} pods should set it to `eth0`.

## CUDA Forward Compatibility

The entrypoint detects host NVIDIA driver versions older than the bundled `cuda-compat` layer and automatically prepends `/usr/local/cuda/compat` to
`LD_LIBRARY_PATH`. No flag or env var needed — the check runs on every container start, then the entrypoint `exec`s the command you passed.

## How We Build

These images are curated builds tracking the [Ray](https://github.com/ray-project/ray) project:

- **Built from upstream releases** — Ray and PyTorch come from upstream wheels, with our own compiled flash-attn / Transformer Engine layered on top
- **Reproducible** — pinned via `pyproject.toml` + `uv.lock`, with build wheels cached in S3 across CI runs
- **Regression-tested** — every release is gated on unit, single-GPU, multi-node EFA, and multi-node KubeRay training tests before publication
- **Security-patched** — continuously maintained with security patches from {{ aws }} on an Amazon Linux 2023 base
