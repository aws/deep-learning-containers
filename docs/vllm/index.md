# LLM Serving using vLLM DLC

Production-ready Docker images for serving large language models with [vLLM](https://docs.vllm.ai/) on {{ aws }}. Built on Amazon Linux 2023 with
ongoing security patching.

## Images

| Platform | Image | Default Port |
| --- | --- | --- |
| EC2 / EKS | `public.ecr.aws/deep-learning-containers/vllm:server-cuda` | 8000 |
| Amazon SageMaker AI | `public.ecr.aws/deep-learning-containers/vllm:server-sagemaker-cuda` | 8080 |

All images are also available on the [ECR Public Gallery](https://gallery.ecr.aws/deep-learning-containers/vllm). For private ECR URIs, see
[Image Access](../get_started/index.md).

## What's Included

In addition to vLLM and its core stack (PyTorch, CUDA 12.9, NCCL, Python 3.12), the images bundle:

- **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)** — fused attention kernels with precompiled cubins for fast cold start
- **[DeepEP](https://github.com/deepseek-ai/DeepEP)** — expert-parallel kernels for large MoE models (DeepSeek, Qwen MoE)
- **[LMCache](https://github.com/LMCache/LMCache) + [NIXL](https://github.com/ai-dynamo/nixl)** — KV-cache offloading and disaggregated prefill/decode
- **[runai-model-streamer](https://github.com/run-ai/runai-model-streamer)** — stream model weights directly from S3, GCS, or Azure
- **[EFA](https://aws.amazon.com/hpc/efa/) and [OpenMPI](https://www.open-mpi.org/)** — high-throughput multi-node networking on supported instances

The SageMaker image additionally includes [standard-supervisor](https://github.com/aws/model-hosting-container-standards) for process auto-recovery,
custom handlers, and dependency installation. See [Amazon SageMaker AI Deployment](deployment/sagemaker.md) for details.

## API Endpoints

The container runs vLLM's [OpenAI-compatible API server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html). Common endpoints:

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/chat/completions` | Chat-style generation |
| `POST /v1/completions` | Legacy text completion |
| `POST /v1/embeddings` | Generate embeddings (embedding models) |
| `POST /v1/audio/transcriptions` | Speech-to-text (ASR models) |
| `POST /v1/responses` | Stateful response API |
| `POST /v1/rerank`, `/v1/score` | Reranking and scoring |
| `GET /v1/models` | List loaded model(s) |
| `POST /tokenize`, `/detokenize` | Tokenizer access |
| `POST /v1/load_lora_adapter`, `/v1/unload_lora_adapter` | Dynamic LoRA management |
| `GET /health`, `/ping` | Liveness probe |
| `GET /metrics` | Prometheus metrics |

Refer to [vLLM's API documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) for request/response schemas and the full
endpoint list.

## How We Build

These images are curated builds, not simple repackages of upstream releases:

- **Built from upstream main** — images track the vLLM main branch with frequent releases, each gated by our regression test suite before publication.
- **Regression-tested** — validated against a suite of models on every release. See [Supported Models](models/index.md) for the full list.
- **Security-patched** — continuously maintained with security patches from {{ aws }} on an Amazon Linux 2023 base.
