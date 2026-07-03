# Changelog

Changelog for the Amazon Linux 2023-based PyTorch images (`2.12-cu130-amzn2023`, `2.12-cpu-amzn2023`, and the corresponding `*-sagemaker` variants).

* * *

## PyTorch 2.12 — 2026-07-02

**Tags:** `2.12-cu130-amzn2023` · `2.12-cpu-amzn2023` · `2.12-cu130-amzn2023-sagemaker` · `2.12-cpu-amzn2023-sagemaker`

**Bundled versions:** PyTorch 2.12.1 · `torchvision` 0.27.1 · `torchaudio` 2.11.0 · CUDA 13.0.2 · Python 3.12 · NCCL 2.26.2 · EFA 1.47.0 · GDRCopy
2.4.4 · flash-attn 2.8.3 · Transformer Engine 2.12.0 · DeepSpeed 0.18.8

### Highlights

- Bumped PyTorch to 2.12.1, with `torchvision` 0.27.1
- 2.12.1 fixes a Triton illegal-memory-access in the `convolution2d_bwd_weight` kernel on B100/B200 (sm100) GPUs
  ([pytorch#187081](https://github.com/pytorch/pytorch/issues/187081))

* * *

## PyTorch 2.11 — 2026-04-30

**Tags:** `2.11-cu130-amzn2023` · `2.11-cpu-amzn2023` · `2.11-cu130-amzn2023-sagemaker` · `2.11-cpu-amzn2023-sagemaker`

### Highlights

- Initial release of PyTorch DLC images on Amazon Linux 2023
- PyTorch 2.11.0 (with `torchvision` 0.26.0 and `torchaudio` 2.11.0)
- CUDA 13.0.2, Python 3.12, NCCL 2.26.2 (GPU variants)
- EFA 1.47.0 with the AWS NCCL OFI plugin and GDRCopy 2.4.4 for multi-node training
- flash-attn 2.8.3 and Transformer Engine 2.12.0 for fused attention and FP8 training
- DeepSpeed 0.18.8 for memory-efficient large-model training
- NCCL `all_reduce_perf` binary at `/usr/local/bin/all_reduce_perf` for verifying EFA connectivity
- Pre-configured OpenSSH server (port 22) for inter-node MPI/`torchrun` launches
- SageMaker variants include the `sagemaker-pytorch-training` toolkit, MLflow, SHAP, smclarify, and SageMaker-specific data libraries
