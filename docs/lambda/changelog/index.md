# Changelog

Changelog for the {{ lambda }} GPU base images (`base`, `cupy`, `pytorch`, `vllm`, `sglang`).

* * *

## v1.0

**Tags:** `base:cuda` · `cupy:cuda` · `pytorch:cuda` · `vllm:cuda` · `sglang:cuda`

### Highlights

- Initial release of the {{ lambda_short }} GPU base image line, for GPU functions on {{ lambda_short }} Managed Instances
- CUDA 13.0.3 on Amazon Linux 2023, Python 3.13, `awslambdaric` 4.2.2
- Five images from one multi-stage Dockerfile: CUDA-only `base`, `cupy`, `pytorch`, and the `vllm` and `sglang` inference servers
- vLLM and SGLang ship a default handler that serves an OpenAI-compatible API with no handler code — set `MODEL_ID` and deploy
- A {{ lambda_short }} extension owns the engine on the vLLM and SGLang images: it loads the model during Init, holds the environment out of service
  until `/health` answers, and recycles it if the engine dies or stops answering
- FlashInfer kernels ship prebuilt for sm80, sm89 and sm120f in the vLLM image, removing the per-environment JIT build from cold start
- Weight streaming from {{ amazon }} S3 straight into GPU memory via the Run:ai Model Streamer (`*_LOAD_FORMAT=runai_streamer`) and
  `s3torchconnector`, with no `/tmp` staging
- FFmpeg 8.0.1 built with NVENC and NVDEC in the `pytorch` image, for GPU transcoding without CUDA code
- All caches redirected to `/tmp`, the only writable path in a {{ lambda_short }} execution environment

### Core Packages

| Package | Version |
| --- | --- |
| CUDA | 13.0.3 |
| Python | 3.13 |
| awslambdaric | 4.2.2 |
| PyTorch | 2.14.1 (cu130) |
| TorchVision | 0.29.1 |
| TorchAudio | 2.11.0 |
| Transformers | 5.17.0 |
| vLLM | 0.33.0 |
| SGLang | 0.5.23+dlc1 |
| SGLang Kernel | 0.4.10 |
| FlashInfer | 0.7.2.post1 (vLLM) |
| CuPy | 14.0.1 (`cupy-cuda13x`) |
| s3torchconnector | 1.5.0 |
| FFmpeg | 8.0.1 |
| NumPy | 2.4.2 |
