---
hide:
  - navigation
  - toc
---

![AWS Logo](assets/logos/AWS_logo_RGB.svg#only-light){: width="30%" }
![AWS Logo](assets/logos/AWS_logo_RGB_REV.svg#only-dark){: width="30%" }
{: style="text-align:center" }

<h1 style="text-align: center;">AWS Deep Learning Containers</h1>

**One stop shop for running AI/ML on AWS**
{: style="text-align:center" }

[**AWS Doc**](https://aws.amazon.com/ai/machine-learning/containers/) · [**Available Images**](reference/available_images.md) · [**Tutorials**](tutorials/index.md)
{: style="text-align:center" }

---

## 🔥 What's New

### 🚀 Release Highlights

- **[2025/12/19]** Released v0.13.0 [vLLM DLC](https://gallery.ecr.aws/deep-learning-containers/vllm)
    - EC2/EKS/ECS: `public.ecr.aws/deep-learning-containers/vllm:0.13-gpu-py312-ec2`
    - SageMaker: `public.ecr.aws/deep-learning-containers/vllm:0.13.0-gpu-py312`
- **[2025/11/17]** Released first [SGLang DLC](https://gallery.ecr.aws/deep-learning-containers/sglang)
    - SageMaker: `public.ecr.aws/deep-learning-containers/sglang:0.5.5-gpu-py312`

### 🎉 Hot Off the Press

- 🌐 **[Master Distributed Training on EKS](https://aws.amazon.com/blogs/machine-learning/configure-and-verify-a-distributed-training-cluster-with-aws-deep-learning-containers-on-amazon-eks/)** - Set up and validate a distributed training environment on Amazon EKS for scalable ML model training across multiple nodes.
- 🔄 **[Level Up with SageMaker AI & MLflow](https://aws.amazon.com/blogs/machine-learning/use-aws-deep-learning-containers-with-amazon-sagemaker-ai-managed-mlflow/)** - Integrate AWS DLCs with Amazon SageMaker's managed MLflow service for streamlined experiment tracking and model management.
- 🚀 **[Deploy LLMs Like a Pro on EKS](https://aws.amazon.com/blogs/architecture/deploy-llms-on-amazon-eks-using-vllm-deep-learning-containers/)** - Deploy and serve Large Language Models efficiently on Amazon EKS using vLLM Deep Learning Containers.
- 🎯 **[Web Automation with Meta Llama 3.2 Vision](https://aws.amazon.com/blogs/machine-learning/fine-tune-and-deploy-meta-llama-3-2-vision-for-generative-ai-powered-web-automation-using-aws-dlcs-amazon-eks-and-amazon-bedrock/)** - Fine-tune and deploy Meta's Llama 3.2 Vision model for AI-powered web automation.
- ⚡ **[Supercharge Your DL Environment](https://aws.amazon.com/blogs/machine-learning/streamline-deep-learning-environments-with-amazon-q-developer-and-mcp/)** - Integrate AWS DLCs with Amazon Q Developer and Model Context Protocol (MCP).

### 🎓 Hands-on Workshop

- 🚀 **[LLM Deployment on EKS Workshop](https://catalog.us-east-1.prod.workshops.aws/workshops/c22b50fb-64b1-4e18-8d0f-ce990f87eed3/en-US)** - Deploy and optimize LLMs on Amazon EKS using vLLM Deep Learning Containers. | [Sample Code](https://github.com/aws-samples/sample-vllm-on-eks-with-dlc)

---

## About

AWS Deep Learning Containers (DLCs) are a suite of Docker images that streamline the deployment of AI/ML workloads on Amazon SageMaker, Amazon EKS, and Amazon EC2.

### 🎯 What We Offer

- **Pre-optimized Environments** - Production-ready containers with optimized deep learning frameworks
- **Latest AI/ML Tools** - Quick access to cutting-edge frameworks like vLLM, SGLang, and PyTorch
- **Multi-Platform Support** - Run seamlessly on SageMaker, EKS, or EC2
- **Enterprise-Ready** - Built with security, performance, and scalability in mind

### 💪 Key Benefits

- **Rapid Deployment** - Get started in minutes with pre-configured environments
- **Framework Flexibility** - Support for popular frameworks like PyTorch, TensorFlow, and more
- **Performance Optimized** - Containers tuned for AWS infrastructure
- **Regular Updates** - Quick access to latest framework releases and security patches
- **AWS Integration** - Seamless compatibility with AWS AI/ML services

### 🎮 Perfect For

- Data Scientists building and training models
- ML Engineers deploying production workloads
- DevOps teams managing ML infrastructure
- Researchers exploring cutting-edge AI capabilities

### 🔒 Security & Compliance

Our containers undergo rigorous security scanning and are regularly updated to address vulnerabilities, ensuring your ML workloads run on a secure foundation.

---

## Quick Links

- [Getting Started](get_started/index.md) - Get up and running in minutes
- [Tutorials](tutorials/index.md) - Step-by-step guides
- [Available Images](reference/available_images.md) - Browse all container images
- [Support Policy](reference/support_policy.md) - Framework versions and timelines

## Getting Help

- [GitHub Issues](https://github.com/aws/deep-learning-containers/issues) - Report bugs or request features
- [AWS Documentation](https://docs.aws.amazon.com/deep-learning-containers/) - Official AWS documentation

## License

This project is licensed under the Apache-2.0 License.
