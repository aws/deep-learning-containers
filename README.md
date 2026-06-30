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

- **[2026/06/29]** [SGLang Server v1.1 (AL2023)](https://gallery.ecr.aws/deep-learning-containers/sglang) — EC2: `server-cuda-v1.1` · SageMaker: `server-sagemaker-cuda-v1.1` · SGLang `0.5.13`; adds NIXL KV connector for prefill/decode disaggregation and `runai-model-streamer[s3,gcs,azure]` for fast weight streaming from object storage; starlette CVE patch.
- **[2026/06/29]** [SGLang v0.5.14](https://gallery.ecr.aws/deep-learning-containers/sglang) — EC2: `0.5.14-gpu-py312-ec2` · SageMaker: `0.5.14-gpu-py312` · GLM 5.2, LiquidAI LFM2.5, Kimi-K2.7-Code, DeepSeek V4 on GB300.
- **[2026/06/14]** [vLLM v0.23.0](https://gallery.ecr.aws/deep-learning-containers/vllm) — EC2: `0.23.0-gpu-py312-ec2` · SageMaker: `0.23.0-gpu-py312` · Step-3.7-Flash, Cosmos3 Reasoner, Gemma 4 Unified (encoder-free), Granite Speech Plus, Cohere Mini Code; Anthropic Messages API structured output.
- **[2026/06/13]** [SGLang v0.5.13](https://gallery.ecr.aws/deep-learning-containers/sglang) — EC2: `0.5.13-gpu-py312-ec2` · SageMaker: `0.5.13-gpu-py312` · DeepSeek V4 (BCG, HiSparse PD, PP+PD), Kimi-K2.5, MiMo-V2, Ideogram 4 (FP8/NVFP4); SM120 + FP4 indexer support.
- **[2026/06/12]** [SGLang Server v1.0 (AL2023)](https://gallery.ecr.aws/deep-learning-containers/sglang) — EC2: `server-cuda-v1.0` · SageMaker: `server-sagemaker-cuda-v1.0` · First Amazon Linux 2023 SGLang Server images, built from upstream source; OpenAI-compatible API (port 30000 EC2/EKS, 8080 SageMaker); CUDA 13.0 for H100 + Blackwell; PyTorch 2.11.0; EFA, DeepEP, and Mooncake KV-cache bundled.
- **[2026/06/05]** [vLLM v0.22.1](https://gallery.ecr.aws/deep-learning-containers/vllm) — EC2: `0.22.1-gpu-py312-ec2` · SageMaker: `0.22.1-gpu-py312` · JetBrains Mellum v2; DeepSeek-V4, OlmoHybrid, HyperCLOVAX fixes; AMD Zen CPU zentorch kernels.
- **[2026/05/30]** [vLLM v0.22.0](https://gallery.ecr.aws/deep-learning-containers/vllm) — EC2: `0.22.0-gpu-py312-ec2` · SageMaker: `0.22.0-gpu-py312` · MiniCPM-V 4.6, InternS2 Preview, OpenVLA, EXAONE-4.5; DeepSeek V4 maturity (NVFP4 fused MoE, MTP speculative decoding); Blackwell SM12x support.

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
