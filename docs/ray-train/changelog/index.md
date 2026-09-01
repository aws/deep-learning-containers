# Changelog

Changelog for the Amazon Linux 2023-based Ray Train DLC images (`train-ml-cuda`).

* * *

## Ray Train v1.1 — 2026-09-01

**Tags:** `train-ml-cuda` · `train-ml-cuda-v1` · `train-ml-cuda-v1.1` · `train-ml-cuda-v1.1.0`

**Bundled versions:** Ray 2.58.0 · PyTorch 2.13.0 · `torchvision` 0.28.0 · CUDA 13.0.2 · Python 3.13 · NCCL 2.29.7 · EFA 1.49.0 · GDRCopy 2.4.4 ·
flash-attn 2.8.3 · Transformer Engine 2.13.0 · DeepSpeed 0.19.2 · PyTorch Lightning 2.6.5 · Transformers 5.13.0

### Changes

- Upgraded EFA from 1.47.0 to 1.49.0 (aws-ofi-nccl 1.20.0)

* * *

## Ray Train v1.0 — 2026-08-26

**Tags:** `train-ml-cuda` · `train-ml-cuda-v1` · `train-ml-cuda-v1.0` · `train-ml-cuda-v1.0.0`

**Bundled versions:** Ray 2.58.0 · PyTorch 2.13.0 · `torchvision` 0.28.0 · CUDA 13.0.2 · Python 3.13 · NCCL 2.29.7 · EFA 1.47.0 · GDRCopy 2.4.4 ·
flash-attn 2.8.3 · Transformer Engine 2.13.0 · DeepSpeed 0.19.2 · PyTorch Lightning 2.6.5 · Transformers 5.13.0

### Highlights

- Initial release of Ray Train DLC images on Amazon Linux 2023 — the PyTorch training stack plus Ray, replacing the upstream `rayproject/ray-ml` image
  deprecated at Ray 2.31.0
- Ray 2.58.0 with the `default`, `train`, `tune`, and `data` extras; `ray[default]` supplies the dashboard and job-submission server used by
  `ray job submit` and KubeRay's health probes
- PyTorch 2.13.0 with `torchvision` 0.28.0 on CUDA 13.0.2 and Python 3.13
- EFA 1.47.0 with OpenMPI, the AWS NCCL OFI plugin, and GDRCopy 2.4.4 for multi-node collectives over EFA
- flash-attn 2.8.3 and Transformer Engine 2.13.0 for fused attention and FP8 training; DeepSpeed 0.19.2 for ZeRO sharding
- PyTorch Lightning 2.6.5 backing Ray Train's `RayFSDPStrategy` and `RayDDPStrategy` integrations, plus Transformers 5.13.0, `datasets` 5.0.0,
  `accelerate` 1.14.0, `evaluate` 0.4.6, and `torchmetrics` 1.9.0
- One image serves both Ray head and Ray worker roles, so KubeRay's `RayCluster` spec drives the pods on {{ eks_short }} and HyperPod-EKS, and
  `ray start` drives them on {{ ec2_short }}
- Ports 6379 (GCS), 8265 (dashboard and job API), and 10001 (Ray Client) exposed
- NCCL `all_reduce_perf` binary at `/usr/local/bin/all_reduce_perf` for verifying EFA connectivity
- Pre-configured OpenSSH server (port 22) for inter-node MPI launches
