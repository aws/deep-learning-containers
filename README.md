<div align="center"> <img src="https://aws.github.io/deep-learning-containers/assets/logos/AWS_logo_RGB.svg" alt="AWS Logo" width="30%"> </div>

<h1 align="center">AWS Deep Learning Containers</h1>

<p align="center"><strong>One stop shop for running AI/ML on AWS</strong></p>

<p align="center"><a href="https://aws.github.io/deep-learning-containers/"><strong>Docs</strong></a> ·
<a href="https://aws.github.io/deep-learning-containers/reference/available_images/"><strong>Available Images</strong></a> · <a href="https://aws.github.io/deep-learning-containers/tutorials/"><strong>Tutorials</strong></a></p>

<p align="center">
  <a href="https://github.com/aws/deep-learning-containers/actions/workflows/pytorch.autorelease-2.14-ec2.yml"><img src="https://github.com/aws/deep-learning-containers/actions/workflows/pytorch.autorelease-2.14-ec2.yml/badge.svg" alt="Auto Release - PyTorch 2.14"></a>
  <a href="https://github.com/aws/deep-learning-containers/actions/workflows/tensorflow-training.autorelease-2.21-sagemaker.yml"><img src="https://github.com/aws/deep-learning-containers/actions/workflows/tensorflow-training.autorelease-2.21-sagemaker.yml/badge.svg" alt="Auto Release - TensorFlow Training 2.21"></a>
  <a href="https://github.com/aws/deep-learning-containers/actions/workflows/tensorflow-inference.autorelease-2.20-sagemaker.yml"><img src="https://github.com/aws/deep-learning-containers/actions/workflows/tensorflow-inference.autorelease-2.20-sagemaker.yml/badge.svg" alt="Auto Release - TensorFlow Inference 2.20"></a>
  <a href="https://github.com/aws/deep-learning-containers/actions/workflows/vllm.autorelease-ec2-amzn2023.yml"><img src="https://github.com/aws/deep-learning-containers/actions/workflows/vllm.autorelease-ec2-amzn2023.yml/badge.svg" alt="Auto Release - vLLM"></a>
  <a href="https://github.com/aws/deep-learning-containers/actions/workflows/vllm-omni.autorelease-ec2.yml"><img src="https://github.com/aws/deep-learning-containers/actions/workflows/vllm-omni.autorelease-ec2.yml/badge.svg" alt="Auto Release - vLLM-Omni"></a>
  <a href="https://github.com/aws/deep-learning-containers/actions/workflows/sglang.autorelease-ec2-amzn2023.yml"><img src="https://github.com/aws/deep-learning-containers/actions/workflows/sglang.autorelease-ec2-amzn2023.yml/badge.svg" alt="Auto Release - SGLang"></a>
  <a href="https://github.com/aws/deep-learning-containers/actions/workflows/ray.autorelease-ec2.yml"><img src="https://github.com/aws/deep-learning-containers/actions/workflows/ray.autorelease-ec2.yml/badge.svg" alt="Auto Release - Ray"></a>
  <a href="https://github.com/aws/deep-learning-containers/actions/workflows/base.autorelease-cu130.yml"><img src="https://github.com/aws/deep-learning-containers/actions/workflows/base.autorelease-cu130.yml/badge.svg" alt="Auto Release - Base cu130"></a>
  <a href="https://github.com/aws/deep-learning-containers/actions/workflows/base.autorelease-cu132.yml"><img src="https://github.com/aws/deep-learning-containers/actions/workflows/base.autorelease-cu132.yml/badge.svg" alt="Auto Release - Base cu132"></a>
</p>

______________________________________________________________________

## About

AWS Deep Learning Containers (DLCs) are pre-built Docker images for running AI/ML workloads on AWS. Each image is tested and patched for security vulnerabilities. For more details, visit our [documentation](https://aws.github.io/deep-learning-containers/).

______________________________________________________________________

## 🔥 What's New

### 🚀 Release Highlights

- **[2026/11/08]** [vLLM Server v2.8 (AL2023)](https://gallery.ecr.aws/deep-learning-containers/vllm) — EC2: `server-cuda-v2.8` · SageMaker: `server-sagemaker-cuda-v2.8` · vLLM `0.33.0` (up from 0.31.1), built from the `v0.33.0` release tag; PyTorch 2.14.0, FlashInfer 0.7.2.post1, Transformers 5.20.2.
- **[2026/10/19]** [SGLang v0.5.22 (Ubuntu)](https://gallery.ecr.aws/deep-learning-containers/sglang) — EC2: `0.5.22-gpu-py312-ec2` · SageMaker: `0.5.22-gpu-py312` · EmbeddingGemma 2, HunyuanImage-3.0-Instruct; per-token NVFP4 MoE for ReLU2 activation; Blackwell dual-GEMM fusion for MLP; decode context parallelism for DSA models.
- **[2026/10/18]** [vLLM v0.32.0 (Ubuntu)](https://gallery.ecr.aws/deep-learning-containers/vllm) — EC2: `0.32.0-gpu-py312-ec2` · SageMaker: `0.32.0-gpu-py312` · New models: EmbeddingGemma 2 omni-modal embeddings, MiniCPM-V 4.7, Nemotron 3.5 ASR; video input in the Transformers backend; LoRA for BERT/RoBERTa embeddings; NVFP4 KV cache on SM8x/SM90/SM12x; new `/v1/systemone` structured-decisions endpoint. Breaking: Python 3.10 dropped, default CORS origins restricted, `transformers>=5.16.1,<5.20.0`.
- **[2026/10/14]** [vLLM Server v2.7 (AL2023)](https://gallery.ecr.aws/deep-learning-containers/vllm) — EC2: `server-cuda-v2.7` · SageMaker: `server-sagemaker-cuda-v2.7` · vLLM `0.31.1` (up from 0.30.0), built from `1a3a7a8`; Reflection Beam (501B total / 23B active MoE) with the `beam` reasoning and tool-call parsers and DFlash2 speculative decoding; FlashInfer 0.7.0.post1, Transformers 5.19.0.
- **[2026/10/09]** [vLLM v0.31.0 (Ubuntu)](https://gallery.ecr.aws/deep-learning-containers/vllm) — EC2: `0.31.0-gpu-py312-ec2` · SageMaker: `0.31.0-gpu-py312` · DeepSeek-V4.1-Flash performance (FlashMLA on SM100); `vllm preload` weight-cache daemon for fast restarts; MoonEP all2all backend; `--max-num-active-seqs`; per-request multimodal kwargs now gated behind `--trust-request-mm-kwargs`.

### 📢 Support Updates

- **[2026/04/28]** We cannot guarantee security patching on Ubuntu-based vLLM and SGLang images due to the lack of Ubuntu Pro licensing. Customers may continue using these images at their own discretion and risk. We recommend migrating to our Amazon Linux-based images.
- **[2026/02/10]** Extended support for PyTorch 2.6 Inference containers until June 30, 2026
  - PyTorch 2.6 Inference images will continue to receive security patches and updates through end of June 2026
  - For complete framework support timelines, see our [Support Policy](https://aws.github.io/deep-learning-containers/reference/support_policy/)

### 📝 Blog Posts

- **[Distributed Training on Amazon EKS](https://aws.amazon.com/blogs/machine-learning/configure-and-verify-a-distributed-training-cluster-with-aws-deep-learning-containers-on-amazon-eks/)** - Configure and validate a distributed training cluster with DLCs on Amazon EKS.
- **[DLCs with Amazon SageMaker AI & MLflow](https://aws.amazon.com/blogs/machine-learning/use-aws-deep-learning-containers-with-amazon-sagemaker-ai-managed-mlflow/)** - Use DLCs with SageMaker AI managed MLflow for experiment tracking and model management.
- **[LLM Serving on Amazon EKS with vLLM](https://aws.amazon.com/blogs/architecture/deploy-llms-on-amazon-eks-using-vllm-deep-learning-containers/)** - Deploy and serve LLMs on Amazon EKS using vLLM DLCs.
- **[TorchServe to Ray Serve DLCs](https://aws.amazon.com/blogs/machine-learning/simplify-and-support-your-torchserve-workloads-using-ray-serve-deep-learning-containers/)** - Migrate TorchServe inference workloads to Ray Serve DLCs — AWS-managed, pre-tested containers without manual CUDA or version maintenance.
- **[Fine-tuning Meta Llama 3.2 Vision](https://aws.amazon.com/blogs/machine-learning/fine-tune-and-deploy-meta-llama-3-2-vision-for-generative-ai-powered-web-automation-using-aws-dlcs-amazon-eks-and-amazon-bedrock/)** - Fine-tune and deploy Llama 3.2 Vision for web automation using DLCs, Amazon EKS, and Amazon Bedrock.
- **[DLCs with Amazon Q Developer and MCP](https://aws.amazon.com/blogs/machine-learning/streamline-deep-learning-environments-with-amazon-q-developer-and-mcp/)** - Streamline deep learning environments with Amazon Q Developer and Model Context Protocol.

### 🎓 Workshop

- **[LLM Deployment on Amazon EKS](https://catalog.us-east-1.prod.workshops.aws/workshops/c22b50fb-64b1-4e18-8d0f-ce990f87eed3/en-US)** - Deploy and optimize LLMs on Amazon EKS using vLLM DLCs.
  See also: [Sample Code](https://github.com/aws-samples/sample-vllm-on-eks-with-dlc)

______________________________________________________________________

## License

This project is licensed under the Apache-2.0 License.
