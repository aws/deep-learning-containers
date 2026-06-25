# LLM Serving using SGLang DLC

Production-ready Docker images for serving large language models with [SGLang](https://docs.sglang.ai/) on {{ aws }}. Built on Amazon Linux 2023 with
ongoing security patching.

## Images

| Platform | Image | Default Port |
| --- | --- | --- |
| {{ ec2_short }} / {{ eks_short }} | `public.ecr.aws/deep-learning-containers/sglang:server-cuda` | 30000 |
| {{ sagemaker }} | `public.ecr.aws/deep-learning-containers/sglang:server-sagemaker-cuda` | 8080 |

All images are also available on the [ECR Public Gallery](https://gallery.ecr.aws/deep-learning-containers/sglang). For private ECR URIs, see
[Image Access](../get_started/index.md).

## What's Included

In addition to SGLang and its core stack (PyTorch 2.11, CUDA 13.0, NCCL, Python 3.12), the images bundle:

- **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)** — fused attention kernels with precompiled cubins for fast cold start
- **[DeepEP](https://github.com/deepseek-ai/DeepEP)** — expert-parallel kernels for large MoE models (DeepSeek, Qwen MoE)
- **[Mooncake](https://github.com/kvcache-ai/Mooncake)** — KV-cache transfer engine for disaggregated prefill/decode
- **[sgl-kernel](https://github.com/sgl-project/sglang/tree/main/sgl-kernel)** — SGLang's custom CUDA kernels, built from source for the bundled CUDA
  arch list
- **[EFA](https://aws.amazon.com/hpc/efa/) and [OpenMPI](https://www.open-mpi.org/)** — high-throughput multi-node networking on supported instances

The images are built from SGLang source against the H100 (sm_90) and Blackwell (sm_100, sm_103) CUDA architectures.

## API Endpoints

The container runs SGLang's [OpenAI-compatible API server](https://docs.sglang.ai/basic_usage/openai_api.html). Common endpoints:

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/chat/completions` | Chat-style generation |
| `POST /v1/completions` | Legacy text completion |
| `POST /v1/embeddings` | Generate embeddings (embedding models) |
| `POST /generate` | SGLang-native generation API |
| `GET /v1/models` | List loaded model(s) |
| `GET /get_model_info` | Model metadata |
| `GET /health`, `/health_generate` | Liveness probe |
| `POST /flush_cache` | Flush the radix attention cache |
| `GET /metrics` | Prometheus metrics |

Refer to [SGLang's API documentation](https://docs.sglang.ai/basic_usage/openai_api.html) for request/response schemas and the full endpoint list.

## How We Build

These images are curated builds, not simple repackages of upstream releases:

- **Built from upstream source** — images build SGLang from a pinned upstream commit, each gated by our regression test suite before publication.
- **Regression-tested** — validated against a suite of models on every release. See [Supported Models](models/index.md) for the full list.
- **Security-patched** — continuously maintained with security patches from {{ aws }} on an Amazon Linux 2023 base.
