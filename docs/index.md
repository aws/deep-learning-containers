---
hide:
  - navigation
  - toc
---
<div align="center"> <img src="assets/logos/AWS_logo_RGB.svg#only-light" alt="AWS Logo" width="30%">
<img src="assets/logos/AWS_logo_RGB_REV.svg#only-dark" alt="AWS Logo" width="30%"> </div>

<h1 align="center">AWS Deep Learning Containers</h1>

<p align="center"><strong>Pre-built Docker images for running AI/ML workloads on AWS.</strong></p> <p align="center">Tested for performance and
patched for security vulnerabilities.</p>

* * *

## What are DLCs?

AWS Deep Learning Containers (DLCs) are Docker images pre-configured with deep learning frameworks and tools. AWS builds, tests, and security-patches
them so you can focus on your workload instead of environment setup.

Each image includes a framework (e.g. vLLM, PyTorch, Ray), its dependencies, and optimized libraries — ready to run on AWS compute services. All DLC
images are provided at no cost — you only pay for the compute resources you use.

## Getting Started

It's easy to get started. For example, to run a large language model server:

```bash
docker pull public.ecr.aws/deep-learning-containers/vllm:server-cuda
docker run --gpus all -p 8000:8000 \
  public.ecr.aws/deep-learning-containers/vllm:server-cuda \
  --model openai/gpt-oss-20b
```

Query the server:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "openai/gpt-oss-20b", "messages": [{"role": "user", "content": "Hello!"}]}'
```

LLM serving is just one example. DLCs cover a range of AI/ML workloads — explore guides by use case:

<div class="grid cards" style="grid-template-columns: repeat(3, 1fr);" markdown>

-   **Serve Large Language Models**

    ---

    Deploy large language models with vLLM or SGLang on EC2, EKS, or Amazon SageMaker AI.

    [vLLM Guide](vllm/index.md) · [SGLang Guide](sglang/index.md)

-   **Serve Multimodal Models**

    ---

    Serve TTS, image generation, video generation, and omni-chat models with vLLM-Omni.

    [vLLM-Omni Guide](vllm-omni/index.md)

-   **Serve ML Models**

    ---

    Deploy any ML model with Ray Serve on EC2 or Amazon SageMaker AI — NLP, vision, audio, and tabular.

    [Ray Guide](ray/index.md)

-   **Train Models**

    ---

    Run distributed training with PyTorch on GPU or CPU, with EFA, NCCL, flash-attn, and DeepSpeed pre-installed.

    [PyTorch Guide](pytorch/index.md)

-   **Train Models with TensorFlow**

    ---

    Run TensorFlow training on Amazon SageMaker AI with EFA-capable multi-node support on Amazon Linux 2023.

    [TensorFlow Guide](tensorflow/index.md)

-   **Build Your Own Image**

    ---

    Use the lightweight Base images (CUDA + Python on Amazon Linux 2023) as the `FROM` for your custom AI/ML container.

    [Base Guide](base/index.md)

</div>

For step-by-step walkthroughs on EKS, SageMaker, and more, see our [blog posts](tutorials/index.md). You can also
[subscribe to release notifications](get_started/release_notifications.md) to stay up to date with new images.

Looking for something else? [Let us know on GitHub](https://github.com/aws/deep-learning-containers/issues).

* * *

<p align="center"> <a href="https://github.com/aws/deep-learning-containers">GitHub</a> ·
<a href="https://gallery.ecr.aws/deep-learning-containers">ECR Gallery</a> · <a href="reference/support_policy/">Support Policy</a> ·
<a href="security/">Security</a> </p>
