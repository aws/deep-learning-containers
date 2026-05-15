# vLLM Inference

Production-ready Docker images for serving large language models with [vLLM](https://docs.vllm.ai/) on {{ aws }}. Built on Amazon Linux 2023 with
ongoing security patching.

## Images

| Platform | Image |
| --- | --- |
| EC2 / EKS | `public.ecr.aws/deep-learning-containers/vllm:server-cuda` |
| Amazon SageMaker AI | `public.ecr.aws/deep-learning-containers/vllm:server-sagemaker-cuda` |

For private ECR URIs, see [Image Access](../get_started/index.md).

## How We Build

These images are curated builds, not simple repackages of upstream releases:

- **Built from upstream main** — images track the vLLM main branch with frequent releases, each gated by our regression test suite before publication.
- **Regression-tested** — validated against a suite of models on every release. See [Supported Models](models/index.md) for the full list.
- **Security-patched** — continuously maintained with security patches from {{ aws }} on an Amazon Linux 2023 base.

Each image includes vLLM (OpenAI-compatible API server), PyTorch, CUDA, and NCCL for multi-GPU inference.
