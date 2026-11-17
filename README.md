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

- **[2026/11/14]** [vLLM-Omni v1.9 (AL2023)](https://gallery.ecr.aws/deep-learning-containers/vllm) — EC2: `omni-cuda-v1.9` · SageMaker: `omni-sagemaker-cuda-v1.9` · vLLM-Omni `0.32.0` (up from 0.29.0rc1) on vLLM 0.32.0; FlashInfer 0.7.0.post1, Transformers 5.19.0, Pillow 13.0.0. Breaking: diffusion models no longer accept `SM_VLLM_ENABLE_LORA` / `SM_VLLM_MAX_LORA_RANK`, and `SM_VLLM_API_KEY` takes a single key.
- **[2026/11/13]** [vLLM Server ARM64 CPU v1.1 (AL2023)](https://gallery.ecr.aws/deep-learning-containers/vllm-arm64) — EC2: `server-cpu-v1.1` · SageMaker: `server-sagemaker-cpu-v1.1` · vLLM `0.33.0` (up from 0.30.0) on AWS Graviton, built from the `v0.33.0` release tag.
- **[2026/11/12]** [NVIDIA Dynamo v1.0 (1.6.2, AL2023)](https://gallery.ecr.aws/deep-learning-containers/dynamo) — SageMaker HyperPod (EKS): `1.6.2-efa-vllm-runtime-v1` · `1.6.2-efa-sglang-runtime-v1` · `1.6.2-efa-trtllm-runtime-v1` (and non-EFA `1.6.2-<backend>-runtime-v1`) · Initial release: NVIDIA Dynamo 1.6.2 built from source with vLLM 0.31.0, SGLang 0.5.21, or TensorRT-LLM 1.3.0rc29, NIXL KV cache transfer over EFA, CUDA 13.0, Python 3.12.
- **[2026/11/10]** [Strands Decider v1.0 (AL2023)](https://gallery.ecr.aws/deep-learning-containers/strands-decider) — SageMaker: `serve-sagemaker-cuda-v1` · Initial release: serve strands-decider decision models (Qwen3.5 2B and Gemma 4 E2B, E4B, 12B and 26B-A4B adapters) that answer yes/no, choice, and score questions with calibrated probabilities; strands-decider 0.3.1.dev1, PyTorch 2.7.1 (CUDA 12.8), Python 3.13.
- **[2026/11/08]** [vLLM Server v2.8 (AL2023)](https://gallery.ecr.aws/deep-learning-containers/vllm) — EC2: `server-cuda-v2.8` · SageMaker: `server-sagemaker-cuda-v2.8` · vLLM `0.33.0` (up from 0.31.1), built from the `v0.33.0` release tag; PyTorch 2.14.0, FlashInfer 0.7.2.post1, Transformers 5.20.2.
- **[2026/11/07]** [vLLM v0.33.0 (Ubuntu)](https://gallery.ecr.aws/deep-learning-containers/vllm) — EC2: `0.33.0-gpu-py312-ec2` · SageMaker: `0.33.0-gpu-py312` · New models: Reflection Beam-1.0 with its DFlash2 drafter, AliceAI-Foundation-80B-A3B, Recursive Flint, Muse Spark Open, Step-5 Preview, NVIDIA Alpamayo 2 Super; PyTorch 2.14.0; XPress refiner and opt-in FLy verification for speculative decoding; `/v1/decisions` on by default. Breaking: Medusa and the MLP speculator removed, NIXL connector needs NIXL 1.5+.
- **[2026/11/07]** [Ray LLM v1.1 (2.60.0, AL2023)](https://gallery.ecr.aws/deep-learning-containers/ray) — EC2/EKS: `serve-llm-cuda-v1.1` · Ray `2.60.0` (up from 2.58.0) and vLLM `0.29.0` (up from 0.26.0); PyTorch 2.13.0, Transformers 5.17.0.
- **[2026/11/05]** [SGLang v0.5.23 (Ubuntu)](https://gallery.ecr.aws/deep-learning-containers/sglang) — EC2: `0.5.23-gpu-py312-ec2` · SageMaker: `0.5.23-gpu-py312` · New models: Muse Spark Open, Beam-1.0, MiniMax-M3.1-Flash, MiniCPM-V 4.7, FLUX.3 Image Klein; up to 1.5x DeepSeek-V4.1 prefill throughput on GB300; opt-in low-latency megakernel decode; Paged Experts for MoE models larger than GPU memory; LoRA on FP8/NVFP4 quantized MoE.
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

- **[OpenFold3 on Amazon SageMaker JumpStart](https://aws.amazon.com/blogs/machine-learning/openfold3-on-amazon-sagemaker-jumpstart-predict-biomolecular-structures/)** - Predict 3D structures of proteins, DNA, RNA, and ligand complexes with OpenFold3-preview on asynchronous SageMaker AI endpoints, including multi-GPU parallelism, bring-your-own MSAs, and scaling to zero.
- **[Distributed Training with Ray Train DLCs and EFA on Amazon EKS](https://aws.amazon.com/blogs/machine-learning/distributed-training-with-ray-train-dlcs-and-efa-on-eks/)** - Fine-tune Qwen2.5-1.5B with PyTorch FSDP across eight GPUs on two EFA-enabled `g6.12xlarge` instances using the Ray Train DLC and KubeRay — no image build required.
- **[Multi-node Inference with Ray LLM DLCs](https://aws.amazon.com/blogs/machine-learning/run-multi-node-inference-workloads-using-ray-llm-deep-learning-containers/)** - Serve one LLM across two GPU nodes on Amazon EKS with separate prefill and decode nodes behind a single OpenAI-compatible endpoint, using NIXL KV-cache transfer over EFA.
- **[Decision Models with llama.cpp on SageMaker AI](https://aws.amazon.com/blogs/machine-learning/deploy-decision-models-with-llama-cpp-on-amazon-sagemaker-ai/)** - Deploy a quantized Qwen3.5-4B to a SageMaker AI real-time endpoint with the llama.cpp DLC, on CPU or GPU, including probabilities over a fixed option set.
- **[Real-time Voice with vLLM-Omni on SageMaker AI (Part 1)](https://aws.amazon.com/blogs/machine-learning/build-real-time-voice-applications-with-vllm-omni-on-sagemaker-ai-part-1/)** - Deploy Qwen3-TTS with the vLLM-Omni DLC and stream generated speech over a persistent bidirectional connection from a Gradio application.
- **[Image and Video Generation with vLLM-Omni on SageMaker AI (Part 2)](https://aws.amazon.com/blogs/machine-learning/generate-images-and-video-with-vllm-omni-on-sagemaker-ai-part-2/)** - Generate an image with FLUX.2-klein through real-time inference, then animate it to video with Wan2.1-VACE through asynchronous inference, from one vLLM-Omni DLC.
- **[Speaker-labeled Transcription with WhisperX on SageMaker AI](https://aws.amazon.com/blogs/machine-learning/speaker-labeled-transcription-with-whisperx-on-sagemaker-ai/)** - Deploy the WhisperX DLC — Whisper, wav2vec2 forced alignment, and speaker diarization — to SageMaker AI real-time and asynchronous endpoints for word-level, speaker-labeled transcription.
- **[TorchServe to Ray Serve DLCs](https://aws.amazon.com/blogs/machine-learning/simplify-and-support-your-torchserve-workloads-using-ray-serve-deep-learning-containers/)** - Migrate TorchServe inference workloads to Ray Serve DLCs — AWS-managed, pre-tested containers without manual CUDA or version maintenance.
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
