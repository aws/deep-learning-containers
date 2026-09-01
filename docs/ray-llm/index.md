# LLM Serving using the Ray LLM DLC

Production-ready Docker image for online LLM serving with [Ray Serve LLM](https://docs.ray.io/en/latest/serve/llm/index.html) and
[vLLM](https://docs.vllm.ai/) on {{ aws }}. Built on Amazon Linux 2023 with ongoing security patching.

This image pairs Ray Serve's HTTP serving layer with the vLLM inference engine, so one tag turns a model into a live LLM endpoint. The
`build_openai_app` helper from the `ray[llm]` extra runs vLLM engines as Ray Serve deployments and exposes OpenAI-compatible endpoints; vLLM supplies
the continuous batching, paged KV-cache, and tensor parallelism. The same image runs on a single GPU on {{ ec2_short }} and scales to multi-node
tensor-parallel serving on
{{ eks }} via [KubeRay](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started.html).

## Images

| Platform | Variant | Image |
| --- | --- | --- |
| {{ ec2_short }} / {{ eks_short }} | GPU | `public.ecr.aws/deep-learning-containers/ray:serve-llm-cuda` |

Ray LLM shares the `ray` repository with the [Ray Serve DLC](../ray/index.md) (`serve-ml` prefix) and the [Ray Train DLC](../ray-train/index.md)
(`train-ml` prefix), using the `serve-llm` tag prefix. The image is also available on the
[ECR Public Gallery](https://gallery.ecr.aws/deep-learning-containers/ray). For private ECR URIs, see [Image Access](../get_started/index.md).

## What's Included

The image bundles the full serving stack so you can stand up an LLM endpoint without building a custom image:

- **[Ray](https://docs.ray.io/) 2.58.0** with the `default`, `serve`, and `llm` extras — `build_openai_app` serves OpenAI-compatible endpoints backed
  by vLLM
- **[vLLM](https://docs.vllm.ai/) 0.26.0** — inference engine with continuous batching, paged KV-cache, and tensor parallelism
- **[PyTorch](https://pytorch.org/) 2.11.0** with `torchvision` 0.26.0 and `torchaudio` 2.11.0 — CUDA 13.0 wheels
- **CUDA 13.0.2** with cuDNN, plus NCCL for multi-GPU collectives
- **[Transformers](https://huggingface.co/docs/transformers) 5.14.1** — tokenizers, processors, and model configs vLLM loads from
- **[FastAPI](https://fastapi.tiangolo.com/) 0.133.1** with Uvicorn (via Ray Serve) — the async HTTP stack
- **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)** — GPU sampling kernels, with CUDA headers pre-installed for JIT
- **[EFA](https://aws.amazon.com/hpc/efa/) with OpenMPI and the AWS NCCL OFI plugin** — for multi-node tensor-parallel serving
- **boto3, awscli** — AWS SDK, for pulling models from S3
- **Python 3.13** — in a venv at `/opt/venv` (`PATH` already set)

For distributed training with Ray Train, use the [Ray Train DLC](../ray-train/index.md). For non-LLM Ray Serve model serving, use the
[Ray Serve DLC](../ray/index.md).

## Ports

| Port | Purpose |
| --- | --- |
| **8000** | Ray Serve HTTP — the OpenAI-compatible endpoint (`/v1/chat/completions`, `/v1/completions`) |
| **8265** | Ray dashboard and job-submission API |
| **6379** | GCS — where workers connect the head in a multi-node cluster |

The image declares ports **6379** and **8265** via `EXPOSE`. The serving port — **8000** by default, set by the Serve config's `http_options.port` —
is not, so publish it yourself: pass `-p 8000:8000` to `docker run` on {{ ec2_short }}, or declare a `containerPort: 8000` in your Kubernetes
manifest. If you change `http_options.port`, publish that port instead.

`FI_PROVIDER=efa` and `NCCL_DEBUG=INFO` are set in the image, and the base image writes `NCCL_SOCKET_IFNAME` exclusions to `/etc/nccl.conf` so NCCL
auto-detects the right interface on any host. Our multi-node EKS test relies on this auto-detection and sets no interface override; set
`NCCL_SOCKET_IFNAME` explicitly only if auto-detection picks the wrong interface on your platform.

## CUDA Forward Compatibility

The entrypoint detects host NVIDIA driver versions older than the bundled `cuda-compat` layer and automatically prepends `/usr/local/cuda/compat` to
`LD_LIBRARY_PATH`. No flag or env var needed — the check runs on every container start, then the entrypoint `exec`s the command you passed.

## How We Build

These images are curated builds tracking the [Ray](https://github.com/ray-project/ray) and [vLLM](https://github.com/vllm-project/vllm) projects:

- **Built from upstream releases** — Ray and vLLM are installed from upstream wheels, each build gated by our test suite before publication.
- **Security-patched** — continuously maintained with security patches from {{ aws }} on an Amazon Linux 2023 base.
