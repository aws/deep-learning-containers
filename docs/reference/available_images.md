# Available Deep Learning Containers Images

Replace `<repository-name>` and `<image-tag>` based on your desired container.

## Region Availability

| Region                | Code           | General | Neuron | Example URL                                                                       |
| --------------------- | -------------- | ------- | ------ | --------------------------------------------------------------------------------- |
| US East (Ohio)        | us-east-2      | ✅      | ✅     | `763104351884.dkr.ecr.us-east-2.amazonaws.com/<repository-name>:<image-tag>`      |
| US East (N. Virginia) | us-east-1      | ✅      | ✅     | `763104351884.dkr.ecr.us-east-1.amazonaws.com/<repository-name>:<image-tag>`      |
| US West (Oregon)      | us-west-2      | ✅      | ✅     | `763104351884.dkr.ecr.us-west-2.amazonaws.com/<repository-name>:<image-tag>`      |
| EU (Ireland)          | eu-west-1      | ✅      | ✅     | `763104351884.dkr.ecr.eu-west-1.amazonaws.com/<repository-name>:<image-tag>`      |
| Asia Pacific (Tokyo)  | ap-northeast-1 | ✅      | ✅     | `763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/<repository-name>:<image-tag>` |

## Authentication

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
```

______________________________________________________________________

## Base Containers

| Framework | Platform    | Python | Example URL                                                                                |
| --------- | ----------- | ------ | ------------------------------------------------------------------------------------------ |
| CUDA 13.0 | EC2/ECS/EKS | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/base:13.0.0-gpu-py312-cu130-ubuntu22.04-ec2` |
| CUDA 12.9 | EC2/ECS/EKS | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/base:12.9.1-gpu-py312-cu129-ubuntu22.04-ec2` |

______________________________________________________________________

## vLLM Containers

| Framework | Platform    | Python | Example URL                                                            |
| --------- | ----------- | ------ | ---------------------------------------------------------------------- |
| vLLM 0.13 | SageMaker   | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/vllm:0.13-gpu-py312`     |
| vLLM 0.13 | EC2/ECS/EKS | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/vllm:0.13-gpu-py312-ec2` |

______________________________________________________________________

## SGLang Containers

| Framework  | Platform  | Python | Example URL                                                         |
| ---------- | --------- | ------ | ------------------------------------------------------------------- |
| SGLang 0.5 | SageMaker | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/sglang:0.5-gpu-py312` |

______________________________________________________________________

## EC2 Framework Containers

| Framework     | Job Type  | CPU/GPU | Python | Example URL                                                                                            |
| ------------- | --------- | ------- | ------ | ------------------------------------------------------------------------------------------------------ |
| PyTorch 2.9.0 | training  | CPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.9.0-cpu-py312-ubuntu22.04-ec2`        |
| PyTorch 2.9.0 | training  | GPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-ec2`  |
| PyTorch 2.6.0 | inference | CPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-cpu-py312-ubuntu22.04-ec2`       |
| PyTorch 2.6.0 | inference | GPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-ec2` |

______________________________________________________________________

## SageMaker Framework Containers

| Framework         | Job Type  | CPU/GPU | Python | Example URL                                                                                                      |
| ----------------- | --------- | ------- | ------ | ---------------------------------------------------------------------------------------------------------------- |
| PyTorch 2.9.0     | training  | GPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker`      |
| PyTorch 2.6.0     | inference | GPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker`     |
| TensorFlow 2.19.0 | training  | GPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.19.0-gpu-py312-cu125-ubuntu22.04-sagemaker`  |
| TensorFlow 2.19.0 | inference | GPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.19.0-gpu-py312-cu122-ubuntu22.04-sagemaker` |

______________________________________________________________________

## Large Model Inference Containers

| Framework                       | Accelerator | Python | Example URL                                                                         |
| ------------------------------- | ----------- | ------ | ----------------------------------------------------------------------------------- |
| DJLServing 0.36.0 + vLLM 0.12.0 | GPU         | py312  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.36.0-lmi18.0.0-cu128` |
| DJLServing 0.35.0 + vLLM 0.11.1 | GPU         | py312  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.35.0-lmi17.0.0-cu128` |

______________________________________________________________________

## Neuron Containers

| Framework     | Neuron SDK | Job Type  | Instance Types | Example URL                                                                                                        |
| ------------- | ---------- | --------- | -------------- | ------------------------------------------------------------------------------------------------------------------ |
| PyTorch 2.8.0 | 2.26.1     | inference | trn1/trn2/inf2 | `763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-neuronx:2.8.0-neuronx-py311-sdk2.26.1-ubuntu22.04` |
| PyTorch 2.8.0 | 2.26.1     | training  | trn1/trn2/inf2 | `763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuronx:2.8.0-neuronx-py311-sdk2.26.1-ubuntu22.04`  |

______________________________________________________________________

## HuggingFace Containers

| Framework                    | Job Type  | CPU/GPU | Example URL                                                                                                                       |
| ---------------------------- | --------- | ------- | --------------------------------------------------------------------------------------------------------------------------------- |
| PyTorch 2.6.0 + transformers | inference | GPU     | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.6.0-transformers4.49.0-gpu-py312-cu124-ubuntu22.04` |
| PyTorch 2.5.1 + transformers | training  | GPU     | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.5.1-transformers4.49.0-gpu-py311-cu124-ubuntu22.04`  |

______________________________________________________________________

## Additional Resources

- [GitHub Releases](https://github.com/aws/deep-learning-containers/releases)
- [ECR Public Gallery](https://gallery.ecr.aws/deep-learning-containers/)
- [Support Policy](support_policy.md)
