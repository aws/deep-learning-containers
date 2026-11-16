# Blog Posts

Hands-on walkthroughs published on the AWS blogs, newest first.

## 2026

- **[OpenFold3 on Amazon SageMaker JumpStart](https://aws.amazon.com/blogs/machine-learning/openfold3-on-amazon-sagemaker-jumpstart-predict-biomolecular-structures/)**
  — predict 3D structures of proteins, DNA, RNA, and ligand complexes with OpenFold3-preview on asynchronous {{ sagemaker }} endpoints, including
  multi-GPU parallelism, bring-your-own MSAs, and scaling to zero. *(13 Nov 2026)*
- **[Distributed training with Ray Train DLCs and EFA on {{ eks }}](https://aws.amazon.com/blogs/machine-learning/distributed-training-with-ray-train-dlcs-and-efa-on-eks/)**
  — fine-tune Qwen2.5-1.5B with PyTorch FSDP across eight GPUs on two EFA-enabled `g6.12xlarge` instances using the
  [Ray Train DLC](../ray-train/index.md) and KubeRay, with no image build. *(28 Oct 2026)*
- **[Multi-node inference with Ray LLM DLCs](https://aws.amazon.com/blogs/machine-learning/run-multi-node-inference-workloads-using-ray-llm-deep-learning-containers/)**
  — serve one LLM across two GPU nodes on {{ eks }} with separate prefill and decode nodes behind a single OpenAI-compatible endpoint, using NIXL KV
  cache transfer over EFA. *(12 Oct 2026)*
- **[Decision models with llama.cpp on {{ sagemaker }}](https://aws.amazon.com/blogs/machine-learning/deploy-decision-models-with-llama-cpp-on-amazon-sagemaker-ai/)**
  — deploy a quantized Qwen3.5-4B to a real-time endpoint with the [llama.cpp DLC](../llama-cpp/index.md) on CPU or GPU, including probabilities over
  a fixed option set. *(12 Oct 2026)*
- **[Real-time voice applications with vLLM-Omni on {{ sagemaker }} — Part 1](https://aws.amazon.com/blogs/machine-learning/build-real-time-voice-applications-with-vllm-omni-on-sagemaker-ai-part-1/)**
  — deploy Qwen3-TTS with the [vLLM-Omni DLC](../vllm-omni/index.md) and stream generated speech over a persistent bidirectional connection from a
  Gradio application. *(28 Sep 2026)*
- **[Image and video generation with vLLM-Omni on {{ sagemaker }} — Part 2](https://aws.amazon.com/blogs/machine-learning/generate-images-and-video-with-vllm-omni-on-sagemaker-ai-part-2/)**
  — generate an image with FLUX.2-klein through real-time inference, then animate it to video with Wan2.1-VACE through asynchronous inference, from
  one [vLLM-Omni DLC](../vllm-omni/index.md). *(28 Sep 2026)*
- **[Speaker-labeled transcription with WhisperX on {{ sagemaker }}](https://aws.amazon.com/blogs/machine-learning/speaker-labeled-transcription-with-whisperx-on-sagemaker-ai/)**
  — deploy the [WhisperX DLC](../whisperx/index.md) — Whisper, wav2vec2 forced alignment, and speaker diarization — to real-time and asynchronous
  endpoints for word-level, speaker-labeled transcription. *(24 Sep 2026)*
- **[Simplify and support your TorchServe workloads using Ray Serve DLCs](https://aws.amazon.com/blogs/machine-learning/simplify-and-support-your-torchserve-workloads-using-ray-serve-deep-learning-containers/)**
  — migrate TorchServe inference workloads to the [Ray Serve DLC](../ray/index.md) and deploy a vision-language model on {{ eks }} without maintaining
  CUDA or framework versions yourself. *(9 Sep 2026)*

## 2025

- **[Configure and verify a distributed training cluster with DLCs on {{ eks }}](https://aws.amazon.com/blogs/machine-learning/configure-and-verify-a-distributed-training-cluster-with-aws-deep-learning-containers-on-amazon-eks/)**
  — configure and validate a multi-node training cluster running DLCs on {{ eks }}. *(15 Oct 2025)*
- **[Use DLCs with {{ sagemaker }} managed MLflow](https://aws.amazon.com/blogs/machine-learning/use-aws-deep-learning-containers-with-amazon-sagemaker-ai-managed-mlflow/)**
  — track experiments and manage models with {{ sagemaker }} managed MLflow from inside a DLC. *(18 Sep 2025)*
- **[Deploy LLMs on {{ eks }} using vLLM DLCs](https://aws.amazon.com/blogs/architecture/deploy-llms-on-amazon-eks-using-vllm-deep-learning-containers/)**
  — deploy and serve LLMs on {{ eks }} with the [vLLM DLC](../vllm/index.md). *(14 Aug 2025)*
- **[Fine-tune and deploy Meta Llama 3.2 Vision](https://aws.amazon.com/blogs/machine-learning/fine-tune-and-deploy-meta-llama-3-2-vision-for-generative-ai-powered-web-automation-using-aws-dlcs-amazon-eks-and-amazon-bedrock/)**
  — fine-tune and deploy Llama 3.2 Vision for generative AI-powered web automation using DLCs, {{ eks }}, and {{ amazon }} Bedrock. *(29 Jul 2025)*
- **[Streamline deep learning environments with {{ amazon }} Q Developer and MCP](https://aws.amazon.com/blogs/machine-learning/streamline-deep-learning-environments-with-amazon-q-developer-and-mcp/)**
  — set up and troubleshoot DLC environments with {{ amazon }} Q Developer and the Model Context Protocol. *(22 Jul 2025)*

## Workshop

- **[LLM deployment on {{ eks }}](https://catalog.us-east-1.prod.workshops.aws/workshops/c22b50fb-64b1-4e18-8d0f-ce990f87eed3/en-US)** — deploy and
  optimize LLMs on {{ eks }} using vLLM DLCs, with [sample code](https://github.com/aws-samples/sample-vllm-on-eks-with-dlc).
