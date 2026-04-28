<div align="center"> <img src="https://aws.github.io/deep-learning-containers/assets/logos/AWS_logo_RGB.svg" alt="AWS Logo" width="30%"> </div>

<h1 align="center">AWS Deep Learning Containers</h1>

<p align="center"><strong>One stop shop for running AI/ML on AWS</strong></p>

<p align="center"><a href="https://aws.github.io/deep-learning-containers/"><strong>Docs</strong></a> ·
<a href="https://aws.github.io/deep-learning-containers/reference/available_images/"><strong>Available Images</strong></a> · <a href="https://aws.github.io/deep-learning-containers/tutorials/"><strong>Tutorials</strong></a></p>

<p align="center">
  <a href="https://github.com/aws/deep-learning-containers/actions/workflows/autorelease-vllm-ec2.yml"><img src="https://github.com/aws/deep-learning-containers/actions/workflows/autorelease-vllm-ec2.yml/badge.svg" alt="Auto Release - vLLM EC2"></a>
  <a href="https://github.com/aws/deep-learning-containers/actions/workflows/autorelease-vllm-sagemaker.yml"><img src="https://github.com/aws/deep-learning-containers/actions/workflows/autorelease-vllm-sagemaker.yml/badge.svg" alt="Auto Release - vLLM SageMaker"></a>
  <a href="https://github.com/aws/deep-learning-containers/actions/workflows/autorelease-vllm-omni.yml"><img src="https://github.com/aws/deep-learning-containers/actions/workflows/autorelease-vllm-omni.yml/badge.svg" alt="Auto Release - vLLM-Omni"></a>
  <a href="https://github.com/aws/deep-learning-containers/actions/workflows/autorelease-ray.yml"><img src="https://github.com/aws/deep-learning-containers/actions/workflows/autorelease-ray.yml/badge.svg" alt="Auto Release - Ray"></a>
  <a href="https://github.com/aws/deep-learning-containers/actions/workflows/autorelease-sglang-ec2.yml"><img src="https://github.com/aws/deep-learning-containers/actions/workflows/autorelease-sglang-ec2.yml/badge.svg" alt="Auto Release - SGLang EC2"></a>
  <a href="https://github.com/aws/deep-learning-containers/actions/workflows/autorelease-sglang-sagemaker.yml"><img src="https://github.com/aws/deep-learning-containers/actions/workflows/autorelease-sglang-sagemaker.yml/badge.svg" alt="Auto Release - SGLang SageMaker"></a>
</p>

______________________________________________________________________

## About

AWS Deep Learning Containers (DLCs) are pre-built Docker images for running AI/ML workloads on AWS. Each image is tested and patched for security vulnerabilities. For more details, visit our [documentation](https://aws.github.io/deep-learning-containers/).

______________________________________________________________________

## 🔥 What's New

### 🚀 Release Highlights

- **[2026/04/20]** [vLLM v0.19.1](https://gallery.ecr.aws/deep-learning-containers/vllm) — EC2: `0.19-gpu-py312-ec2` · SageMaker: `0.19-gpu-py312` · This upgrades Transformers to 5.5.4, enabling Gemma 4 support.
- **[2026/04/07]** [SGLang v0.5.10](https://gallery.ecr.aws/deep-learning-containers/sglang) — EC2: `0.5.10-gpu-py312-ec2` · SageMaker: `0.5.10-gpu-py312`
- **[2026/04/07]** [vLLM v0.19.0](https://gallery.ecr.aws/deep-learning-containers/vllm) — EC2: `0.19-gpu-py312-ec2` · SageMaker: `0.19-gpu-py312`
- **[2026/03/26]** [vLLM v0.18.0](https://gallery.ecr.aws/deep-learning-containers/vllm) — EC2: `0.18-gpu-py312-ec2` · SageMaker: `0.18-gpu-py312`
- **[2026/03/23]** [PyTorch Training v2.10.0](https://gallery.ecr.aws/deep-learning-containers/pytorch-training) — EC2: `2.10.0-gpu-py313-cu130-ubuntu22.04-ec2` · SageMaker: `2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker`

### 📢 Support Updates

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
