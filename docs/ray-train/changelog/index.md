# Changelog

Changelog for the Amazon Linux 2023-based Ray Train DLC images (`train-ml-cuda`).

* * *

## v1.0.0 — Unreleased

**Tags:** `train-ml-cuda` · `train-ml-cuda-v1` · `train-ml-cuda-v1.0`

**Bundled versions:** Ray 2.56.0 · PyTorch 2.13.0 · `torchvision` 0.28.0 · CUDA 13.0.2 · Python 3.13 · NCCL 2.29.7 · EFA 1.47.0 · GDRCopy 2.4.4 ·
flash-attn 2.8.3 · Transformer Engine 2.13.0 · DeepSpeed 0.19.2 · PyTorch Lightning 2.6.5 · Transformers 5.13.0

### Highlights

- Initial release of the Ray Train DLC on Amazon Linux 2023 — a multi-node, multi-GPU distributed training image, replacing the upstream
  `rayproject/ray-ml` image deprecated at Ray 2.31.0
- Ray 2.56.0 with the `default`, `train`, `tune`, and `data` extras. `ray[default]` supplies the dashboard and job-submission server that KubeRay
  health probes and `ray job submit` require; `ray[serve]` is deliberately excluded to keep the image training-scoped
- PyTorch 2.13.0 with `torchvision` 0.28.0 on CUDA 13.0.2 and Python 3.13
- EFA 1.47.0 with OpenMPI, the AWS NCCL OFI plugin, and GDRCopy 2.4.4 for multi-node collectives over EFA
- flash-attn 2.8.3 and Transformer Engine 2.13.0 for fused attention and FP8 training; DeepSpeed 0.19.2 for ZeRO sharding
- PyTorch Lightning 2.6.5 for Ray Train's `RayFSDPStrategy` / `RayDDPStrategy` integrations, plus Transformers 5.13.0, `datasets` 5.0.0, `accelerate`
  1.14.0, `evaluate` 0.4.6, and `torchmetrics` 1.9.0
- One image serves both Ray head and Ray worker roles — no head/worker entrypoint is baked in, so KubeRay's `RayCluster` spec drives the pods
- Ports 6379 (GCS), 8265 (dashboard and job API), and 10001 (Ray Client) exposed; port 8000 is not, since Ray Serve is absent
- NCCL `all_reduce_perf` binary at `/usr/local/bin/all_reduce_perf` for verifying EFA connectivity before a long run
- `NCCL_SOCKET_IFNAME` left unpinned in favor of the `^docker0,lo` exclusion default in `/etc/nccl.conf`, so the same image auto-detects the right NIC
  on both {{ eks_short }} pods and {{ ec2_short }} hosts
- Pre-configured OpenSSH server (port 22) for MPI-based multi-node launches
