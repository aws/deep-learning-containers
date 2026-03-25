---
hide:
  - navigation
  - toc
---
<div align="center"> <img src="assets/logos/AWS_logo_RGB.svg#only-light" alt="AWS Logo" width="30%">
<img src="assets/logos/AWS_logo_RGB_REV.svg#only-dark" alt="AWS Logo" width="30%"> </div>

<h1 align="center">AWS Deep Learning Containers</h1>

Pre-built Docker images for running AI/ML workloads on AWS. Deep Learning Containers provide optimized environments for training and serving models on
Amazon SageMaker AI, Amazon EKS, and Amazon EC2 — with the latest frameworks, security patches, and AWS integrations built in.

[Browse on ECR Public Gallery](https://gallery.ecr.aws/?searchTerm=deep+learning+containers) · [Get Started →](get_started/)

* * *

<div class="grid cards" markdown>

- **LLM**

    ---

    Serving · Supported Models

    Serve large language models with vLLM, SGLang, and llama.cpp.

    [Explore LLM →](frameworks/llm.md)

- **ML**

    ---

    Training · Serving

    Train with PyTorch and serve with Ray Serve.

    [Explore ML →](frameworks/ml.md)

</div>

* * *

## What's New

### Release Highlights

- **[2026/03/10]** Released v0.17.0 [vLLM DLCs](https://gallery.ecr.aws/deep-learning-containers/vllm)
  - EC2/EKS/ECS: `public.ecr.aws/deep-learning-containers/vllm:0.17-gpu-py312-ec2`
  - SageMaker: `public.ecr.aws/deep-learning-containers/vllm:0.17-gpu-py312`
- **[2026/03/09]** Released v0.5.9 [SGLang DLCs](https://gallery.ecr.aws/deep-learning-containers/sglang)
  - EC2/EKS/ECS: `public.ecr.aws/deep-learning-containers/sglang:0.5.9-gpu-py312-ec2`
  - SageMaker: `public.ecr.aws/deep-learning-containers/sglang:0.5.9-gpu-py312`
- **[2026/03/09]** Released v0.16.0 [vLLM DLCs](https://gallery.ecr.aws/deep-learning-containers/vllm)
  - EC2/EKS/ECS: `public.ecr.aws/deep-learning-containers/vllm:0.16-gpu-py312-ec2`
  - SageMaker: `public.ecr.aws/deep-learning-containers/vllm:0.16-gpu-py312`
- **[2025/11/17]** Released first [SGLang DLCs](https://gallery.ecr.aws/deep-learning-containers/sglang)
  - SageMaker: `public.ecr.aws/deep-learning-containers/sglang:0.5.5-gpu-py312`

### Support Updates

- **[2026/02/10]** Extended support for PyTorch 2.6 Inference containers until June 30, 2026
  - PyTorch 2.6 Inference images will continue to receive security patches and updates through end of June 2026
  - For complete framework support timelines, see our [Support Policy](reference/support_policy.md)

* * *

## Getting Help

- [GitHub Issues](https://github.com/aws/deep-learning-containers/issues) — Report bugs or request features
- [GitHub Discussions](https://github.com/aws/deep-learning-containers/discussions) — Ask questions and share ideas

## License

This project is licensed under the Apache-2.0 License.
