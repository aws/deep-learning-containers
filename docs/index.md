______________________________________________________________________

hide:

- navigation
- toc

______________________________________________________________________

![AWS Logo](assets/logos/AWS_logo_RGB.svg#only-light){: width="30%" }
![AWS Logo](assets/logos/AWS_logo_RGB_REV.svg#only-dark){: width="30%" }
{: style="text-align:center" }

<h1 style="text-align: center;">{{ dlc_long }}</h1>

**One stop shop for running AI/ML on {{ aws }}**
{: style="text-align:center" }

[**{{ aws }} Doc**](https://aws.amazon.com/ai/machine-learning/containers/) · [**Available Images**](reference/available_images.md) · [**Tutorials**](tutorials/index.md)
{: style="text-align:center" }

______________________________________________________________________

## 🔥 What's New

### 🚀 Release Highlights

- **[2025/12/19]** Released v0.13.0 [vLLM {{ dlc_short }}](https://gallery.ecr.aws/deep-learning-containers/vllm)
  - EC2/EKS/ECS: `public.ecr.aws/deep-learning-containers/vllm:0.13-gpu-py312-ec2`
  - SageMaker: `public.ecr.aws/deep-learning-containers/vllm:0.13.0-gpu-py312`
- **[2025/11/17]** Released first [SGLang {{ dlc_short }}](https://gallery.ecr.aws/deep-learning-containers/sglang)
  - SageMaker: `public.ecr.aws/deep-learning-containers/sglang:0.5.5-gpu-py312`

### 🎉 Hot Off the Press

- 🌐 **[Master Distributed Training on {{ eks }}](https://aws.amazon.com/blogs/machine-learning/configure-and-verify-a-distributed-training-cluster-with-aws-deep-learning-containers-on-amazon-eks/)** - Set up and validate a distributed training environment on {{ eks }} for scalable ML model training across multiple nodes.
- 🔄 **[Level Up with {{ sagemaker }} & MLflow](https://aws.amazon.com/blogs/machine-learning/use-aws-deep-learning-containers-with-amazon-sagemaker-ai-managed-mlflow/)** - Integrate {{ aws }} {{ dlc_short }} with {{ sagemaker }}'s managed MLflow service for streamlined experiment tracking and model management.
- 🚀 **[Deploy LLMs Like a Pro on {{ eks }}](https://aws.amazon.com/blogs/architecture/deploy-llms-on-amazon-eks-using-vllm-deep-learning-containers/)** - Deploy and serve Large Language Models efficiently on {{ eks }} using vLLM {{ dlc }}.
- 🎯 **[Web Automation with Meta Llama 3.2 Vision](https://aws.amazon.com/blogs/machine-learning/fine-tune-and-deploy-meta-llama-3-2-vision-for-generative-ai-powered-web-automation-using-aws-dlcs-amazon-eks-and-amazon-bedrock/)** - Fine-tune and deploy Meta's Llama 3.2 Vision model for AI-powered web automation.
- ⚡ **[Supercharge Your DL Environment](https://aws.amazon.com/blogs/machine-learning/streamline-deep-learning-environments-with-amazon-q-developer-and-mcp/)** - Integrate {{ aws }} {{ dlc_short }} with {{ amazon }} Q Developer and Model Context Protocol (MCP).

### 🎓 Hands-on Workshop

- 🚀 **[LLM Deployment on {{ eks }} Workshop](https://catalog.us-east-1.prod.workshops.aws/workshops/c22b50fb-64b1-4e18-8d0f-ce990f87eed3/en-US)** - Deploy and optimize LLMs on {{ eks }} using vLLM {{ dlc }}. For more information, see [Sample Code](https://github.com/aws-samples/sample-vllm-on-eks-with-dlc)

______________________________________________________________________

## About

{{ dlc_long }} ({{ dlc_short }}) are a suite of Docker images that streamline the deployment of AI/ML workloads on {{ sagemaker }}, {{ eks }}, and {{ ec2 }}.

### 🎯 What We Offer

- **Pre-optimized Environments** - Production-ready containers with optimized deep learning frameworks
- **Latest AI/ML Tools** - Quick access to cutting-edge frameworks like vLLM, SGLang, and PyTorch
- **Multi-Platform Support** - Run seamlessly on {{ sagemaker }}, {{ eks }}, or {{ ec2 }}
- **Enterprise-Ready** - Built with security, performance, and scalability in mind

### 💪 Key Benefits

- **Rapid Deployment** - Get started in minutes with pre-configured environments
- **Framework Flexibility** - Support for popular frameworks like PyTorch, TensorFlow, and more
- **Performance Optimized** - Containers tuned for {{ aws }} infrastructure
- **Regular Updates** - Quick access to latest framework releases and security patches
- **{{ aws }} Integration** - Seamless compatibility with {{ aws }} AI/ML services

### 🎮 Perfect For

- Data Scientists building and training models
- ML Engineers deploying production workloads
- DevOps teams managing ML infrastructure
- Researchers exploring cutting-edge AI capabilities

### 🔒 Security & Compliance

Our containers undergo rigorous security scanning and are regularly updated to address vulnerabilities, ensuring your ML workloads run on a secure foundation.

For more information on our security policy, see [Security](security/index.md).

______________________________________________________________________

## Quick Links

- [Getting Started](get_started/index.md) - Get up and running in minutes
- [Tutorials](tutorials/index.md) - Step-by-step guides
- [Available Images](reference/available_images.md) - Browse all container images
- [Support Policy](reference/support_policy.md) - Framework versions and timelines
- [Security](security/index.md) - Security policy

## Getting Help

- [GitHub Issues](https://github.com/aws/deep-learning-containers/issues) - Report bugs or request features

## License

This project is licensed under the Apache-2.0 License.
