# Distributed Training using Ray Train DLC

Production-ready Docker image for **multi-node, multi-GPU distributed training** with [Ray Train](https://docs.ray.io/en/latest/train/train.html) on
{{ aws }}. Built on Amazon Linux 2023 with ongoing security patching.

The upstream `rayproject/ray-ml` image — the training-oriented Ray image — was deprecated at Ray 2.31.0, leaving no official Ray *training* container.
This DLC fills that gap: it bundles the same interconnect stack as the PyTorch training DLC (EFA, NCCL, GDRCopy, OpenMPI, flash-attn, Transformer
Engine, DeepSpeed) with Ray Train, Tune, and Data, so a single image runs as either a Ray head or a Ray worker on {{ eks }} (via KubeRay), SageMaker
HyperPod-EKS, or plain {{ ec2_short }}.

## Images

One GPU image covers every role and target — KubeRay injects the `ray start` command per pod, so the image does not hard-code a head or worker
entrypoint.

| Platform | Variant | Image | Ports |
| --- | --- | --- | --- |
| {{ ec2_short }} / {{ eks_short }} (KubeRay) | GPU (CUDA) | `public.ecr.aws/deep-learning-containers/ray:train-ml-cuda` | 6379, 8265, 10001 |

Ray Train shares the `ray` repository with the [Ray Serve DLC](../ray/index.md) and is distinguished by the `train-ml` tag prefix. Pinned tags are
published alongside the floating one — `train-ml-cuda-v1`, `train-ml-cuda-v1.0`, and `train-ml-cuda-v1.0.0` — so pin to whichever level of stability
you want. The image is also available on the [ECR Public Gallery](https://gallery.ecr.aws/deep-learning-containers/ray). For private ECR URIs, see
[Image Access](../get_started/index.md).

This image is training-scoped: `ray[serve]` is deliberately **not** installed and port 8000 is not exposed. For model serving with Ray Serve, use the
[Ray Serve DLC](../ray/index.md) instead.

## What's Included

The image layers Ray on top of the full distributed-training stack, so you can launch multi-node training without building a custom image:

- **[Ray](https://docs.ray.io/) 2.58.0** with the `default`, `train`, `tune`, and `data` extras — `ray[default]` is what provides the dashboard and
  the job-submission server that KubeRay's health probes and `ray job submit` depend on
- **[PyTorch](https://pytorch.org/) 2.13.0** with `torchvision` 0.28.0 (CUDA 13.0 wheels)
- **CUDA 13.0.2** with cuDNN and **NCCL 2.29.7** for multi-GPU collectives
- **[EFA](https://aws.amazon.com/hpc/efa/) 1.47.0** with **OpenMPI** and the **AWS NCCL OFI plugin** for low-latency multi-node communication on
  EFA-capable instances
- **[GDRCopy](https://github.com/NVIDIA/gdrcopy) 2.4.4** userspace library for direct GPU-to-NIC memory copies
- **[flash-attn](https://github.com/Dao-AILab/flash-attention) 2.8.3** — fused attention kernels for transformer training
- **[Transformer Engine](https://github.com/NVIDIA/TransformerEngine) 2.13.0** — FP8/BF16 mixed-precision primitives
- **[DeepSpeed](https://www.deepspeed.ai/) 0.19.2** — ZeRO sharding, pipeline parallel, and memory-efficient optimizers
- **[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) 2.6.5** — used by Ray Train's `RayFSDPStrategy` / `RayDDPStrategy` integrations
- **[Transformers](https://huggingface.co/docs/transformers) 5.13.0**, `datasets` 5.0.0, `accelerate` 1.14.0, `evaluate` 0.4.6, `torchmetrics` 1.9.0 —
  the Hugging Face training ecosystem
- **scikit-learn, NumPy, pandas** — feature engineering and tabular workloads
- **boto3, awscli** — AWS SDK pre-installed
- **NCCL test utility** — `all_reduce_perf` at `/usr/local/bin/all_reduce_perf` for verifying EFA/NCCL connectivity before spending GPU-hours
- **OpenSSH** server pre-configured (port 22) for MPI launches between nodes
- **Python 3.13** in a venv at `/opt/venv` (`PATH` already set)

`torchaudio` is intentionally omitted — there is no CUDA 13.0 wheel for it past 2.11.0, and audio I/O is not part of the distributed-training path.

## Ports

| Port | Purpose |
| --- | --- |
| 6379 | Ray GCS (head) — workers join with `--address=<head>:6379` |
| 8265 | Ray Dashboard and job-submission API (`ray job submit --address http://<head>:8265`) |
| 10001 | Ray Client server |
| 22 | OpenSSH, for MPI-based multi-node launches |

## Verified Workloads

Every release is gated on the following tiers before publication:

| Tier | Substrate | What it proves |
| --- | --- | --- |
| Unit | CPU | Ray Train/Tune/Data, PyTorch, Lightning, DeepSpeed, and HF packages import; `ray[serve]` is absent |
| Single GPU | 1 GPU | A real `TorchTrainer` + `ScalingConfig` run converges and a checkpoint round-trips |
| Multi-node EFA | 2 × EFA GPU instances | Cross-node NCCL all-reduce over EFA |
| Multi-node EKS | 2 × 4-GPU nodes via KubeRay | BERT FSDP fine-tune on GLUE/CoLA across 8 GPUs, asserted on convergence |

The multi-node EKS test — a `RayCluster` manifest plus a Ray Train + Lightning FSDP script — doubles as a working reference:
[test/ray-train/eks](https://github.com/aws/deep-learning-containers/tree/main/test/ray-train/eks).

## How We Build

This image is a curated build tracking the [Ray](https://github.com/ray-project/ray) project:

- **Built from upstream releases** — Ray and PyTorch come from upstream wheels, with flash-attn and Transformer Engine compiled in dedicated builder
  stages and cached in S3 across CI runs.
- **Reproducible** — every Python dependency is pinned via `pyproject.toml` + a committed `uv.lock`, installed with `uv sync --frozen`.
- **Security-patched** — continuously maintained with security patches from {{ aws }} on an Amazon Linux 2023 base.
