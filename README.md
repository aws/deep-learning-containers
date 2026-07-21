<div align="center"> <img src="https://aws.github.io/deep-learning-containers/assets/logos/AWS_logo_RGB.svg" alt="AWS Logo" width="30%"> </div>

<h1 align="center">AWS Deep Learning Containers</h1>

<p align="center"><strong>One stop shop for running AI/ML on AWS</strong></p>

<p align="center"><a href="https://aws.github.io/deep-learning-containers/"><strong>Docs</strong></a> ·
<a href="https://aws.github.io/deep-learning-containers/reference/available_images/"><strong>Available Images</strong></a> · <a href="https://aws.github.io/deep-learning-containers/tutorials/"><strong>Tutorials</strong></a></p>

<p align="center">
  <a href="https://github.com/aws/deep-learning-containers/actions/workflows/pytorch.autorelease-2.13-ec2.yml"><img src="https://github.com/aws/deep-learning-containers/actions/workflows/pytorch.autorelease-2.13-ec2.yml/badge.svg" alt="Auto Release - PyTorch 2.13"></a>
  <a href="https://github.com/aws/deep-learning-containers/actions/workflows/tensorflow.autorelease-2.21-sagemaker.yml"><img src="https://github.com/aws/deep-learning-containers/actions/workflows/tensorflow.autorelease-2.21-sagemaker.yml/badge.svg" alt="Auto Release - TensorFlow 2.21"></a>
  <a href="https://github.com/aws/deep-learning-containers/actions/workflows/vllm.autorelease-ec2-amzn2023.yml"><img src="https://github.com/aws/deep-learning-containers/actions/workflows/vllm.autorelease-ec2-amzn2023.yml/badge.svg" alt="Auto Release - vLLM"></a>
  <a href="https://github.com/aws/deep-learning-containers/actions/workflows/vllm-omni.autorelease-ec2.yml"><img src="https://github.com/aws/deep-learning-containers/actions/workflows/vllm-omni.autorelease-ec2.yml/badge.svg" alt="Auto Release - vLLM-Omni"></a>
  <a href="https://github.com/aws/deep-learning-containers/actions/workflows/sglang.autorelease-ec2-amzn2023.yml"><img src="https://github.com/aws/deep-learning-containers/actions/workflows/sglang.autorelease-ec2-amzn2023.yml/badge.svg" alt="Auto Release - SGLang"></a>
  <a href="https://github.com/aws/deep-learning-containers/actions/workflows/ray.autorelease-ec2.yml"><img src="https://github.com/aws/deep-learning-containers/actions/workflows/ray.autorelease-ec2.yml/badge.svg" alt="Auto Release - Ray"></a>
  <a href="https://github.com/aws/deep-learning-containers/actions/workflows/base.autorelease-cu130.yml"><img src="https://github.com/aws/deep-learning-containers/actions/workflows/base.autorelease-cu130.yml/badge.svg" alt="Auto Release - Base cu130"></a>
</p>

______________________________________________________________________

## About

AWS Deep Learning Containers (DLCs) are pre-built Docker images for running AI/ML workloads on AWS. Each image is tested and patched for security vulnerabilities. For more details, visit our [documentation](https://aws.github.io/deep-learning-containers/).

______________________________________________________________________

## 🔥 What's New

### 🚀 Release Highlights

- **[2026/07/20]** [PyTorch v2.13.0](https://gallery.ecr.aws/deep-learning-containers/pytorch) — EC2: `2.13-cu133-amzn2023` · SageMaker: `2.13-cu133-amzn2023-sagemaker` · Amazon Linux 2023 with EFA, flash-attn, and Transformer Engine; PyTorch 2.13.0 with CUDA 13.3.0, NCCL 2.30.7, TE 2.17.0, DeepSpeed 0.19.2.
- **[2026/07/15]** [vLLM v0.25.1 (Ubuntu)](https://gallery.ecr.aws/deep-learning-containers/vllm) — EC2: `0.25.1-gpu-py312-ec2` · SageMaker: `0.25.1-gpu-py312` · Patch release: defer TorchCodec FFmpeg import error to runtime (unblocks startup without system FFmpeg); guard mixed-dtype allreduce RMSNorm quant fusions (fixes NVFP4 garbage output).
- **[2026/07/13]** [vLLM v0.25.0 (Ubuntu)](https://gallery.ecr.aws/deep-learning-containers/vllm) — EC2: `0.25.0-gpu-py312-ec2` · SageMaker: `0.25.0-gpu-py312` · LLaVA-OneVision-2, Unlimited OCR, MOSS-Transcribe-Diarize, openai/privacy-filter, Hy3.
- **[2026/07/11]** [SGLang v0.5.15 (Ubuntu)](https://gallery.ecr.aws/deep-learning-containers/sglang) — EC2: `0.5.15-gpu-py312-ec2` · SageMaker: `0.5.15-gpu-py312` · GLM 5.2 Tuned, Hy3, HRM-Text, LocateAnything-3B.
- **[2026/07/10]** [TensorFlow v2.21.0 (SageMaker training)](https://gallery.ecr.aws/deep-learning-containers/tensorflow-training) — SageMaker CPU: `2.21.0-cpu-py312-amzn2023-sagemaker` · SageMaker GPU: `2.21.0-gpu-py312-cu129-amzn2023-sagemaker` · Amazon Linux 2023 with Python 3.12; GPU images ship CUDA 12.9.1.
- **[2026/07/06]** [SGLang Server v1.2 (AL2023)](https://gallery.ecr.aws/deep-learning-containers/sglang) — EC2: `server-cuda-v1.2` · SageMaker: `server-sagemaker-cuda-v1.2` · SGLang `0.5.14`; adds support for the NVIDIA LocateAnything-3B vision-grounding model and upgrades sgl-kernel 0.4.4, FlashInfer 0.6.12, Mooncake 0.3.11.post1, NCCL 2.30.4, and EFA 1.49.0.
- **[2026/07/02]** [vLLM Server v2.1 (AL2023)](https://gallery.ecr.aws/deep-learning-containers/vllm) — EC2: `server-cuda-v2.1` · SageMaker: `server-sagemaker-cuda-v2.1` · vLLM `0.24.0`; adds JetBrains Mellum2-12B-A2.5B-Thinking (`MellumForCausalLM`); FlashInfer 0.6.12; drops the `transformers<5.10` pin (now requires transformers ≥ 5.5.3).
- **[2026/07/02]** [vLLM-Omni v1.4 (AL2023)](https://gallery.ecr.aws/deep-learning-containers/vllm) — EC2: `omni-cuda-v1.4` · SageMaker: `omni-sagemaker-cuda-v1.4` · SageMaker `/v1/videos` and `/v1/videos/sync` accept `application/json` again (JSON→multipart conversion restored); no framework bump (vLLM-Omni 0.21.0rc1).

### 📢 Support Updates

- **[2026/04/28]** We cannot guarantee security patching on Ubuntu-based vLLM and SGLang images due to the lack of Ubuntu Pro licensing. Customers may continue using these images at their own discretion and risk. We recommend migrating to our Amazon Linux-based images.
- **[2026/02/10]** Extended support for PyTorch 2.6 Inference containers until June 30, 2026
  - PyTorch 2.6 Inference images will continue to receive security patches and updates through end of June 2026
  - For complete framework support timelines, see our [Support Policy](https://aws.github.io/deep-learning-containers/reference/support_policy/)

### 📝 Blog Posts

- **[Distributed Training on Amazon EKS](https://aws.amazon.com/blogs/machine-learning/configure-and-verify-a-distributed-training-cluster-with-aws-deep-learning-containers-on-amazon-eks/)** - Configure and validate a distributed training cluster with DLCs on Amazon EKS.
- **[DLCs with Amazon SageMaker AI & MLflow](https://aws.amazon.com/blogs/machine-learning/use-aws-deep-learning-containers-with-amazon-sagemaker-ai-managed-mlflow/)** - Use DLCs with SageMaker AI managed MLflow for experiment tracking and model management.
- **[LLM Serving on Amazon EKS with vLLM](https://aws.amazon.com/blogs/architecture/deploy-llms-on-amazon-eks-using-vllm-deep-learning-containers/)** - Deploy and serve LLMs on Amazon EKS using vLLM DLCs.
- **[Fine-tuning Meta Llama 3.2 Vision](https://aws.amazon.com/blogs/machine-learning/fine-tune-and-deploy-meta-llama-3-2-vision-for-generative-ai-powered-web-automation-using-aws-dlcs-amazon-eks-and-amazon-bedrock/)** - Fine-tune and deploy Llama 3.2 Vision for web automation using DLCs, Amazon EKS, and Amazon Bedrock.
- **[DLCs with Amazon Q Developer and MCP](https://aws.amazon.com/blogs/machine-learning/streamline-deep-learning-environments-with-amazon-q-developer-and-mcp/)** - Streamline deep learning environments with Amazon Q Developer and Model Context Protocol.

### 🎓 Workshop

- **[LLM Deployment on Amazon EKS](https://catalog.us-east-1.prod.workshops.aws/workshops/c22b50fb-64b1-4e18-8d0f-ce990f87eed3/en-US)** - Deploy and optimize LLMs on Amazon EKS using vLLM DLCs.
  See also: [Sample Code](https://github.com/aws-samples/sample-vllm-on-eks-with-dlc)

______________________________________________________________________

## License

This project is licensed under the Apache-2.0 License.
