# Changelog

Changelog for the {{ lambda }} GPU base images (`base`, `cupy`, `pytorch`, `vllm`, `sglang`).

* * *

## v1.0 — 2026-10-19

**Tags:** `base:cuda` · `cupy:cuda` · `pytorch:cuda` · `vllm:cuda` · `sglang:cuda`

### Highlights

- Initial release of the {{ lambda_short }} GPU base image line, for GPU functions on {{ lambda_short }} Managed Instances
- CUDA 13.0.3 on Amazon Linux 2023, Python 3.13, `awslambdaric` 4.1.0 with pre-fork hook support
- Five images from one multi-stage Dockerfile: CUDA-only `base`, `cupy`, `pytorch`, and the `vllm` and `sglang` inference servers
- vLLM and SGLang ship a default handler that serves an OpenAI-compatible API with no handler code — set `MODEL_ID` and deploy
- Weight streaming from {{ amazon }} S3 straight into GPU memory via the Run:ai Model Streamer (`*_LOAD_FORMAT=runai_streamer`) and
  `s3torchconnector`, with no `/tmp` staging
- FFmpeg 8.0.1 built with NVENC and NVDEC in the `pytorch` image, for GPU transcoding without CUDA code
- All caches redirected to `/tmp`, the only writable path in a {{ lambda_short }} execution environment

### Core Packages

| Package | Version |
| --- | --- |
| CUDA | 13.0.3 |
| Python | 3.13 |
| awslambdaric | 4.1.0 |
| PyTorch | 2.13.0 (cu130) |
| TorchVision | 0.28.0 |
| TorchAudio | 2.11.0 |
| Transformers | 5.17.0 |
| vLLM | 0.30.0 |
| SGLang | 0.5.18+dlc1 |
| SGLang Kernel | 0.4.6.post1 |
| FlashInfer | 0.6.18.post1 (vLLM) |
| CuPy | 14.0.1 (`cupy-cuda13x`) |
| s3torchconnector | 1.5.0 |
| FFmpeg | 8.0.1 |
| NumPy | 2.4.2 |
