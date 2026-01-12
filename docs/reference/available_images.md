# Available Deep Learning Containers Images

Replace `<repository-name>` and `<image-tag>` based on your desired container.

## Getting Started

Once you've selected your desired Deep Learning Containers image, continue with one of the following:

- [Amazon EC2 Tutorials](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-ec2.html)
- [Amazon ECS Tutorials](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-ecs.html)
- [Amazon EKS Tutorials](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-eks.html)
- [Hugging Face on AWS](https://huggingface.co/docs/sagemaker/en/index)
- [Security in AWS Deep Learning Containers](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/security.html)
- [Release Notes](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/dlc-release-notes.html)

## Region Availability

| Region                    | Code           | General | Neuron | Example URL                                                                          |
| ------------------------- | -------------- | ------- | ------ | ------------------------------------------------------------------------------------ |
| US East (Ohio)            | us-east-2      | ✅      | ✅     | `763104351884.dkr.ecr.us-east-2.amazonaws.com/<repository-name>:<image-tag>`         |
| US East (N. Virginia)     | us-east-1      | ✅      | ✅     | `763104351884.dkr.ecr.us-east-1.amazonaws.com/<repository-name>:<image-tag>`         |
| US West (N. California)   | us-west-1      | ✅      | ❌     | `763104351884.dkr.ecr.us-west-1.amazonaws.com/<repository-name>:<image-tag>`         |
| US West (Oregon)          | us-west-2      | ✅      | ✅     | `763104351884.dkr.ecr.us-west-2.amazonaws.com/<repository-name>:<image-tag>`         |
| Africa (Cape Town)        | af-south-1     | ✅      | ❌     | `626614931356.dkr.ecr.af-south-1.amazonaws.com/<repository-name>:<image-tag>`        |
| Asia Pacific (Hong Kong)  | ap-east-1      | ✅      | ❌     | `871362719292.dkr.ecr.ap-east-1.amazonaws.com/<repository-name>:<image-tag>`         |
| Asia Pacific (Hyderabad)  | ap-south-2     | ✅      | ❌     | `772153158452.dkr.ecr.ap-south-2.amazonaws.com/<repository-name>:<image-tag>`        |
| Asia Pacific (Jakarta)    | ap-southeast-3 | ✅      | ❌     | `907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/<repository-name>:<image-tag>`    |
| Asia Pacific (Malaysia)   | ap-southeast-5 | ✅      | ❌     | `550225433462.dkr.ecr.ap-southeast-5.amazonaws.com/<repository-name>:<image-tag>`    |
| Asia Pacific (Melbourne)  | ap-southeast-4 | ✅      | ❌     | `457447274322.dkr.ecr.ap-southeast-4.amazonaws.com/<repository-name>:<image-tag>`    |
| Asia Pacific (Mumbai)     | ap-south-1     | ✅      | ✅     | `763104351884.dkr.ecr.ap-south-1.amazonaws.com/<repository-name>:<image-tag>`        |
| Asia Pacific (Osaka)      | ap-northeast-3 | ✅      | ❌     | `364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/<repository-name>:<image-tag>`    |
| Asia Pacific (Seoul)      | ap-northeast-2 | ✅      | ❌     | `763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/<repository-name>:<image-tag>`    |
| Asia Pacific (Singapore)  | ap-southeast-1 | ✅      | ✅     | `763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/<repository-name>:<image-tag>`    |
| Asia Pacific (Sydney)     | ap-southeast-2 | ✅      | ✅     | `763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/<repository-name>:<image-tag>`    |
| Asia Pacific (Taipei)     | ap-east-2      | ✅      | ✅     | `763104351884.dkr.ecr.ap-east-2.amazonaws.com/<repository-name>:<image-tag>`         |
| Asia Pacific (Thailand)   | ap-southeast-7 | ✅      | ❌     | `590183813437.dkr.ecr.ap-southeast-7.amazonaws.com/<repository-name>:<image-tag>`    |
| Asia Pacific (Tokyo)      | ap-northeast-1 | ✅      | ✅     | `763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/<repository-name>:<image-tag>`    |
| Canada (Central)          | ca-central-1   | ✅      | ❌     | `763104351884.dkr.ecr.ca-central-1.amazonaws.com/<repository-name>:<image-tag>`      |
| Canada (Calgary)          | ca-west-1      | ✅      | ❌     | `204538143572.dkr.ecr.ca-west-1.amazonaws.com/<repository-name>:<image-tag>`         |
| EU (Frankfurt)            | eu-central-1   | ✅      | ✅     | `763104351884.dkr.ecr.eu-central-1.amazonaws.com/<repository-name>:<image-tag>`      |
| EU (Ireland)              | eu-west-1      | ✅      | ✅     | `763104351884.dkr.ecr.eu-west-1.amazonaws.com/<repository-name>:<image-tag>`         |
| EU (London)               | eu-west-2      | ✅      | ❌     | `763104351884.dkr.ecr.eu-west-2.amazonaws.com/<repository-name>:<image-tag>`         |
| EU (Milan)                | eu-south-1     | ✅      | ❌     | `692866216735.dkr.ecr.eu-south-1.amazonaws.com/<repository-name>:<image-tag>`        |
| EU (Paris)                | eu-west-3      | ✅      | ✅     | `763104351884.dkr.ecr.eu-west-3.amazonaws.com/<repository-name>:<image-tag>`         |
| EU (Spain)                | eu-south-2     | ✅      | ❌     | `503227376785.dkr.ecr.eu-south-2.amazonaws.com/<repository-name>:<image-tag>`        |
| EU (Stockholm)            | eu-north-1     | ✅      | ❌     | `763104351884.dkr.ecr.eu-north-1.amazonaws.com/<repository-name>:<image-tag>`        |
| EU (Zurich)               | eu-central-2   | ✅      | ❌     | `380420809688.dkr.ecr.eu-central-2.amazonaws.com/<repository-name>:<image-tag>`      |
| Israel (Tel Aviv)         | il-central-1   | ✅      | ❌     | `780543022126.dkr.ecr.il-central-1.amazonaws.com/<repository-name>:<image-tag>`      |
| Mexico (Central)          | mx-central-1   | ✅      | ❌     | `637423239942.dkr.ecr.mx-central-1.amazonaws.com/<repository-name>:<image-tag>`      |
| Middle East (Bahrain)     | me-south-1     | ✅      | ❌     | `217643126080.dkr.ecr.me-south-1.amazonaws.com/<repository-name>:<image-tag>`        |
| Middle East (UAE)         | me-central-1   | ✅      | ❌     | `914824155844.dkr.ecr.me-central-1.amazonaws.com/<repository-name>:<image-tag>`      |
| South America (Sao Paulo) | sa-east-1      | ✅      | ✅     | `763104351884.dkr.ecr.sa-east-1.amazonaws.com/<repository-name>:<image-tag>`         |
| China (Beijing)           | cn-north-1     | ✅      | ❌     | `727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/<repository-name>:<image-tag>`     |
| China (Ningxia)           | cn-northwest-1 | ✅      | ❌     | `727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/<repository-name>:<image-tag>` |

## Authentication

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
```

Then pull images:

```bash
docker pull <name of container image>
```

## Image Tag Guide

Select your framework, job type (`training`, `inference`, or `general`), and Python version (`py38`, `py39`, `py310`, `py311`, or `py312`).

Pin versions by adding the version tag:

```
763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.4.1-cpu-py37-ubuntu18.04-v1.0
```

---

## Base Containers

| Framework | Platform    | Python | Example URL                                                                                |
| --------- | ----------- | ------ | ------------------------------------------------------------------------------------------ |
| CUDA 13.0 | EC2/ECS/EKS | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/base:13.0.0-gpu-py312-cu130-ubuntu22.04-ec2` |
| CUDA 12.9 | EC2/ECS/EKS | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/base:12.9.1-gpu-py312-cu129-ubuntu22.04-ec2` |
| CUDA 12.8 | EC2/ECS/EKS | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/base:12.8.0-gpu-py312-cu128-ubuntu22.04-ec2` |
| CUDA 12.8 | EC2/ECS/EKS | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/base:12.8.1-gpu-py312-cu128-ubuntu24.04-ec2` |

Also available: [ECR Public Gallery](https://gallery.ecr.aws/deep-learning-containers/base)

---

## vLLM Containers

| Framework | Platform    | Python | Example URL                                                            |
| --------- | ----------- | ------ | ---------------------------------------------------------------------- |
| vLLM 0.13 | SageMaker   | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/vllm:0.13-gpu-py312`     |
| vLLM 0.13 | EC2/ECS/EKS | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/vllm:0.13-gpu-py312-ec2` |

Also available: [ECR Public Gallery](https://gallery.ecr.aws/deep-learning-containers/vllm)

---

## SGLang Containers

| Framework  | Platform  | Python | Example URL                                                         |
| ---------- | --------- | ------ | ------------------------------------------------------------------- |
| SGLang 0.5 | SageMaker | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/sglang:0.5-gpu-py312` |

Also available: [ECR Public Gallery](https://gallery.ecr.aws/deep-learning-containers/sglang)

---

## EC2 Framework Containers

EC2, ECS, and EKS support only.

| Framework     | Job Type  | CPU/GPU | Python | Example URL                                                                                            |
| ------------- | --------- | ------- | ------ | ------------------------------------------------------------------------------------------------------ |
| PyTorch 2.9.0 | training  | CPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.9.0-cpu-py312-ubuntu22.04-ec2`        |
| PyTorch 2.9.0 | training  | GPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-ec2`  |
| PyTorch 2.6.0 | inference | CPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-cpu-py312-ubuntu22.04-ec2`       |
| PyTorch 2.6.0 | inference | GPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-ec2` |

---

## SageMaker Framework Containers

| Framework         | Job Type  | CPU/GPU | Python | Example URL                                                                                                      |
| ----------------- | --------- | ------- | ------ | ---------------------------------------------------------------------------------------------------------------- |
| PyTorch 2.9.0     | training  | CPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.9.0-cpu-py312-ubuntu22.04-sagemaker`            |
| PyTorch 2.9.0     | training  | GPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker`      |
| PyTorch 2.6.0     | inference | CPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-cpu-py312-ubuntu22.04-sagemaker`           |
| PyTorch 2.6.0     | inference | GPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker`     |
| TensorFlow 2.19.0 | training  | CPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.19.0-cpu-py312-ubuntu22.04-sagemaker`        |
| TensorFlow 2.19.0 | training  | GPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.19.0-gpu-py312-cu125-ubuntu22.04-sagemaker`  |
| TensorFlow 2.19.0 | inference | CPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.19.0-cpu-py312-ubuntu22.04-sagemaker`       |
| TensorFlow 2.19.0 | inference | GPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.19.0-gpu-py312-cu122-ubuntu22.04-sagemaker` |

---

## ARM64/Graviton Containers

!!! note

    Starting with PyTorch 2.5, Graviton DLCs are renamed to ARM64 DLCs. The ECR repository name is now `pytorch-inference-arm64` instead of `pytorch-inference-graviton`. They are functionally equivalent.

### EC2 ARM64 Containers

| Framework     | Job Type  | CPU/GPU | Python | Example URL                                                                                                  |
| ------------- | --------- | ------- | ------ | ------------------------------------------------------------------------------------------------------------ |
| PyTorch 2.7.0 | training  | GPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training-arm64:2.7.0-gpu-py312-cu128-ubuntu22.04-ec2`  |
| PyTorch 2.6.0 | inference | CPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-arm64:2.6.0-cpu-py312-ubuntu22.04-ec2`       |
| PyTorch 2.6.0 | inference | GPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-arm64:2.6.0-gpu-py312-cu124-ubuntu22.04-ec2` |

### SageMaker ARM64 Containers

| Framework     | Job Type  | CPU/GPU | Python | Example URL                                                                                                  |
| ------------- | --------- | ------- | ------ | ------------------------------------------------------------------------------------------------------------ |
| PyTorch 2.6.0 | inference | CPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-arm64:2.6.0-cpu-py312-ubuntu22.04-sagemaker` |

---

## NVIDIA Triton Inference Containers

SageMaker support only.

### Versions 23.12+

Available versions: `25.09`, `25.04`, `24.09`, `24.05`, `24.03`, `24.01`, `23.12`

```python
from sagemaker import image_uris

triton_framework = "sagemaker-tritonserver"
region = "us-west-2"
instance_type = "ml.g5.12xlarge"

available_versions = list(image_uris.config_for_framework(triton_framework)["versions"].keys())
image_uri = image_uris.retrieve(
    framework=triton_framework,
    region=region,
    instance_type=instance_type,
    version=available_versions[0],
)
```

See [NVIDIA Triton Release Notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html) for package versions.

### Versions prior to 23.12

| Framework               | Job Type  | CPU/GPU | Python | Example URL                                                                           |
| ----------------------- | --------- | ------- | ------ | ------------------------------------------------------------------------------------- |
| NVIDIA Triton 23.`<XY>` | inference | GPU     | py38   | `007439368137.dkr.ecr.us-east-2.amazonaws.com/sagemaker-tritonserver:23.<XY>-py3`     |
| NVIDIA Triton 23.`<XY>` | inference | CPU     | py38   | `007439368137.dkr.ecr.us-east-2.amazonaws.com/sagemaker-tritonserver:23.<XY>-py3-cpu` |

Available 23.x versions: `23.01, 23.02, 23.03, 23.05, 23.06, 23.07, 23.08, 23.09, 23.10`

Available 22.x versions: `22.05, 22.07, 22.08, 22.09, 22.10, 22.12`

Available 21.x versions: `21.08`

!!! note

    - TensorFlow 1 not supported from version 23.05+
    - FasterTransformer backend not included from version 23.06+

---

## Large Model Inference Containers

!!! note

    Starting LMI V10 (0.28.0), the name changed from LMI DeepSpeed DLC to LMI. DeepSpeed integration is discontinued. Use vLLM or LMI-dist instead. See [deprecation guide](https://github.com/deepjavalibrary/djl-serving/blob/master/serving/docs/lmi/announcements/deepspeed-deprecation.md).

| Framework                                  | Accelerator | Python | Example URL                                                                                 |
| ------------------------------------------ | ----------- | ------ | ------------------------------------------------------------------------------------------- |
| DJLServing 0.36.0 + vLLM 0.12.0            | GPU         | py312  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.36.0-lmi18.0.0-cu128`         |
| DJLServing 0.35.0 + vLLM 0.11.1            | GPU         | py312  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.35.0-lmi17.0.0-cu128`         |
| DJLServing 0.34.0 + vLLM 0.10.2            | GPU         | py312  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.34.0-lmi16.0.0-cu128`         |
| DJLServing 0.33.0 + vLLM 0.8.4             | GPU         | py312  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.33.0-lmi15.0.0-cu128`         |
| DJLServing 0.33.0 + TensorRT-LLM 0.21.0rc1 | GPU         | py312  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.33.0-tensorrtllm0.21.0-cu128` |
| DJLServing 0.32.0 + LMI Dist 13.0.0        | GPU         | py311  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.32.0-lmi14.0.0-cu126`         |
| DJLServing 0.32.0 + TensorRT-LLM 0.12.0    | GPU         | py310  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.32.0-tensorrtllm0.12.0-cu125` |
| DJLServing 0.31.0 + LMI Dist 13.0.0        | GPU         | py311  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.31.0-lmi13.0.0-cu124`         |
| DJLServing 0.30.0 + LMI Dist 12.0.0        | GPU         | py310  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.30.0-lmi12.0.0-cu124`         |
| DJLServing 0.30.0 + TensorRT-LLM 0.12.0    | GPU         | py310  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.30.0-tensorrtllm0.12.0-cu125` |
| DJLServing 0.30.0 + Neuron SDK 2.20.1      | Neuron      | py310  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.30.0-neuronx-sdk2.20.1`       |
| DJLServing 0.29.0 + TensorRT-LLM 0.11.0    | GPU         | py310  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.29.0-tensorrtllm0.11.0-cu124` |
| DJLServing 0.29.0 + LMI Dist 11.0.0        | GPU         | py310  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.29.0-lmi11.0.0-cu124`         |
| DJLServing 0.29.0 + Neuron SDK 2.19.1      | Neuron      | py310  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.29.0-neuronx-sdk2.19.1`       |
| DJLServing 0.28.0 + TensorRT-LLM 0.9.0     | GPU         | py310  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.28.0-tensorrtllm0.9.0-cu122`  |
| DJLServing 0.28.0 + LMI Dist 0.10.0        | GPU         | py310  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.28.0-lmi10.0.0-cu124`         |
| DJLServing 0.28.0 + Neuron SDK 2.18.2      | Neuron      | py310  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.28.0-neuronx-sdk2.18.2`       |

---

## DJL CPU Full Inference Containers

| Framework         | CPU/GPU | Python | Example URL                                                                  |
| ----------------- | ------- | ------ | ---------------------------------------------------------------------------- |
| DJLServing 0.36.0 | CPU     | py311  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.36.0-cpu-full` |
| DJLServing 0.35.0 | CPU     | py311  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.35.0-cpu-full` |
| DJLServing 0.29.0 | CPU     | py310  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.29.0-cpu-full` |
| DJLServing 0.28.0 | CPU     | py310  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.28.0-cpu-full` |
| DJLServing 0.27.0 | CPU     | py310  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.27.0-cpu-full` |

---

## AutoGluon Containers

### Training

| Framework       | CPU/GPU | Python | Example URL                                                                                         |
| --------------- | ------- | ------ | --------------------------------------------------------------------------------------------------- |
| AutoGluon 1.4.0 | GPU     | py311  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/autogluon-training:1.4.0-gpu-py311-cu124-ubuntu22.04` |
| AutoGluon 1.4.0 | CPU     | py311  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/autogluon-training:1.4.0-cpu-py311-ubuntu22.04`       |

### Inference

| Framework       | CPU/GPU | Python | Example URL                                                                                          |
| --------------- | ------- | ------ | ---------------------------------------------------------------------------------------------------- |
| AutoGluon 1.4.0 | GPU     | py311  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/autogluon-inference:1.4.0-gpu-py311-cu124-ubuntu22.04` |
| AutoGluon 1.4.0 | CPU     | py311  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/autogluon-inference:1.4.0-cpu-py311-ubuntu22.04`       |

---

## HuggingFace Containers

### Training

See [GPU Release Page](https://github.com/aws/deep-learning-containers/releases?q=huggingface-pytorch-training+AND+NOT+neuronx&expanded=true) and [HuggingFace documentation](https://huggingface.co/docs/sagemaker/en/dlcs/available#training).

| Framework                       | CPU/GPU | Python | Example URL                                                                                                                        |
| ------------------------------- | ------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| PyTorch 2.5.1 + transformers    | GPU     | py311  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.5.1-transformers4.49.0-gpu-py311-cu124-ubuntu22.04`   |
| PyTorch 2.1.0 + transformers    | GPU     | py310  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.1.0-transformers4.36.0-gpu-py310-cu121-ubuntu20.04`   |
| PyTorch 2.0.0 + transformers    | GPU     | py310  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04`   |
| PyTorch 1.13.1 + transformers   | GPU     | py39   | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04`   |
| TensorFlow 2.6.3 + transformers | GPU     | py38   | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-tensorflow-training:2.6.3-transformers4.17.0-gpu-py38-cu112-ubuntu20.04` |

### Inference

See [GPU and CPU Release Page](https://github.com/aws/deep-learning-containers/releases?q=huggingface-pytorch-inference+AND+NOT+tgi+AND+NOT+neuronx&expanded=true) and [HuggingFace documentation](https://huggingface.co/docs/sagemaker/en/dlcs/available#pytorch-inference-dlc).

| Framework                        | CPU/GPU | Python | Example URL                                                                                                                          |
| -------------------------------- | ------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| PyTorch 2.6.0 + transformers     | CPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.6.0-transformers4.49.0-cpu-py312-ubuntu22.04`          |
| PyTorch 2.6.0 + transformers     | GPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.6.0-transformers4.49.0-gpu-py312-cu124-ubuntu22.04`    |
| PyTorch 2.1.0 + transformers     | CPU     | py310  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.1.0-transformers4.37.0-cpu-py310-ubuntu22.04`          |
| PyTorch 2.1.0 + transformers     | GPU     | py310  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.1.0-transformers4.37.0-gpu-py310-cu118-ubuntu20.04`    |
| PyTorch 2.0.0 + transformers     | CPU     | py310  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-cpu-py310-ubuntu20.04`          |
| PyTorch 2.0.0 + transformers     | GPU     | py310  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04`    |
| PyTorch 1.13.1 + transformers    | CPU     | py39   | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04`          |
| PyTorch 1.13.1 + transformers    | GPU     | py39   | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04`    |
| TensorFlow 2.11.1 + transformers | CPU     | py39   | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-tensorflow-inference:2.11.1-transformers4.26.0-cpu-py39-ubuntu20.04`       |
| TensorFlow 2.11.1 + transformers | GPU     | py39   | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-tensorflow-inference:2.11.1-transformers4.26.0-gpu-py39-cu112-ubuntu20.04` |

### Text Generation Inference (TGI)

See [GPU Release Page](https://github.com/aws/deep-learning-containers/releases?q=tgi+AND+gpu&expanded=true) and [HuggingFace documentation](https://huggingface.co/docs/sagemaker/en/dlcs/available#llm-tgi).

### Text Embeddings Inference (TEI)

See [HuggingFace documentation](https://huggingface.co/docs/sagemaker/dlcs/available#text-embedding-inference).

---

## HuggingFace Neuron Containers

### Neuron TGI

See [NeuronX Release Page](https://github.com/aws/deep-learning-containers/releases?q=tgi+AND+neuronx&expanded=true) and [HuggingFace documentation](https://huggingface.co/docs/optimum-neuron/en/containers#available-optimum-neuron-containers).

### Neuron Inference

See [NeuronX Release Page](https://github.com/aws/deep-learning-containers/releases?q=huggingface-pytorch-inference-neuronx+AND+NOT+tgi&expanded=true) and [HuggingFace documentation](https://huggingface.co/docs/optimum-neuron/en/containers#available-optimum-neuron-containers).

| Framework                     | Neuron SDK | Instance Type | Python | Example URL                                                                                                                                        |
| ----------------------------- | ---------- | ------------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| PyTorch 2.8.0 + transformers  | 2.26.0     | inf2/trn1     | py310  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference-neuronx:2.8.0-transformers4.55.4-neuronx-py310-sdk2.26.0-ubuntu22.04`  |
| PyTorch 2.7.1 + transformers  | 2.24.1     | inf2/trn1     | py310  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference-neuronx:2.7.1-transformers4.51.3-neuronx-py310-sdk2.24.1-ubuntu22.04`  |
| PyTorch 2.1.2 + transformers  | 2.20.0     | inf2/trn1     | py310  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference-neuronx:2.1.2-transformers4.43.2-neuronx-py310-sdk2.20.0-ubuntu20.04`  |
| PyTorch 2.1.2 + transformers  | 2.18.0     | inf2/trn1     | py310  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference-neuronx:2.1.2-transformers4.36.2-neuronx-py310-sdk2.18.0-ubuntu20.04`  |
| PyTorch 1.13.1 + transformers | 2.15.0     | inf2/trn1     | py310  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference-neuronx:1.13.1-transformers4.34.1-neuronx-py310-sdk2.15.0-ubuntu20.04` |
| PyTorch 1.10.2 + transformers | 1.19.1     | inf1          | py37   | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference-neuron:1.10.2-transformers4.20.1-neuron-py37-sdk1.19.1-ubuntu18.04`    |

### Neuron Training

See [NeuronX Release Page](https://github.com/aws/deep-learning-containers/releases?q=huggingface-pytorch-training-neuronx+AND+NOT+tgi&expanded=true) and [HuggingFace documentation](https://huggingface.co/docs/optimum-neuron/en/containers#available-optimum-neuron-containers).

| Framework                     | Neuron SDK | Instance Type | Python | Example URL                                                                                                                                       |
| ----------------------------- | ---------- | ------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| PyTorch 2.8.0 + transformers  | 2.26.0     | trn1          | py310  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training-neuronx:2.8.0-transformers4.55.4-neuronx-py310-sdk2.26.0-ubuntu22.04`  |
| PyTorch 2.7.0 + transformers  | 2.24.1     | trn1          | py310  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training-neuronx:2.7.0-transformers4.51.0-neuronx-py310-sdk2.24.1-ubuntu22.04`  |
| PyTorch 2.1.2 + transformers  | 2.20.0     | trn1          | py310  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training-neuronx:2.1.2-transformers4.48.1-neuronx-py310-sdk2.20.0-ubuntu20.04`  |
| PyTorch 1.13.1 + transformers | 2.18.0     | trn1          | py310  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training-neuronx:1.13.1-transformers4.36.2-neuronx-py310-sdk2.18.0-ubuntu20.04` |

---

## StabilityAI Inference Containers

| Framework           | CPU/GPU | Python | Example URL                                                                                                                       |
| ------------------- | ------- | ------ | --------------------------------------------------------------------------------------------------------------------------------- |
| PyTorch 2.0.1 + SGM | GPU     | py310  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/stabilityai-pytorch-inference:2.0.1-sgm0.1.0-gpu-py310-cu118-ubuntu20.04-sagemaker` |

---

## SageMaker Training Compiler Containers

| Framework                                                | CPU/GPU | Python | Example URL                                                                                                                             |
| -------------------------------------------------------- | ------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| TensorFlow 2.10.0                                        | GPU     | py39   | `763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.10.0-gpu-py39-cu112-ubuntu20.04-sagemaker`                          |
| PyTorch 1.13.1 + Training Compiler                       | GPU     | py39   | `763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-trcomp-training:1.13.1-gpu-py39-cu117-ubuntu20.04-sagemaker`                      |
| PyTorch 1.11.0 + transformers 4.21.1 + Training Compiler | GPU     | py38   | `763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-trcomp-training:1.11.0-transformers4.21.1-gpu-py38-cu113-ubuntu20.04` |

---

## Neuron Containers

!!! note

    Starting from Neuron SDK 2.17.0, Dockerfiles are at [aws-neuron/deep-learning-containers](https://github.com/aws-neuron/deep-learning-containers).

| Framework                                                                                                                                          | Neuron Package                                                    | Neuron SDK | Job Type  | Instance Types | Python | Example URL                                                                                                            |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | ---------- | --------- | -------------- | ------ | ---------------------------------------------------------------------------------------------------------------------- |
| [PyTorch 2.8.0](https://github.com/aws-neuron/deep-learning-containers/blob/2.26.1/pytorch/inference/2.8.0/Dockerfile.neuronx)                     | torch-neuronx, neuronx_distributed, neuronx_distributed_inference | 2.26.1     | inference | trn1/trn2/inf2 | py311  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-neuronx:2.8.0-neuronx-py311-sdk2.26.1-ubuntu22.04`     |
| [PyTorch 2.8.0](https://github.com/aws-neuron/deep-learning-containers/blob/2.26.1/pytorch/training/2.8.0/Dockerfile.neuronx)                      | torch-neuronx, neuronx_distributed, neuronx_distributed_training  | 2.26.1     | training  | trn1/trn2/inf2 | py311  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuronx:2.8.0-neuronx-py311-sdk2.26.1-ubuntu22.04`      |
| [PyTorch 2.7.0](https://github.com/aws-neuron/deep-learning-containers/blob/2.25.0/docker/pytorch/inference/2.7.0/Dockerfile.neuronx)              | torch-neuronx, transformers-neuronx, neuronx_distributed          | 2.25.0     | inference | trn1/trn2/inf2 | py310  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-neuronx:2.7.0-neuronx-py310-sdk2.25.0-ubuntu22.04`     |
| [PyTorch 2.7.0](https://github.com/aws-neuron/deep-learning-containers/blob/2.25.0/docker/pytorch/training/2.7.0/Dockerfile.neuronx)               | torch-neuronx, transformers-neuronx, neuronx_distributed          | 2.25.0     | training  | trn1/trn2/inf2 | py310  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuronx:2.7.0-neuronx-py310-sdk2.25.0-ubuntu22.04`      |
| [PyTorch 2.7.0](https://github.com/aws-neuron/deep-learning-containers/blob/2.24.1/docker/pytorch/inference/2.7.0/Dockerfile.neuronx)              | torch-neuronx, transformers-neuronx, neuronx_distributed          | 2.24.1     | inference | trn1/trn2/inf2 | py310  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-neuronx:2.7.0-neuronx-py310-sdk2.24.1-ubuntu22.04`     |
| [PyTorch 2.7.0](https://github.com/aws-neuron/deep-learning-containers/blob/2.24.1/docker/pytorch/training/2.7.0/Dockerfile.neuronx)               | torch-neuronx, transformers-neuronx, neuronx_distributed          | 2.24.1     | training  | trn1/trn2/inf2 | py310  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuronx:2.7.0-neuronx-py310-sdk2.24.1-ubuntu22.04`      |
| [PyTorch 2.6.0](https://github.com/aws-neuron/deep-learning-containers/blob/2.23.0/docker/pytorch/inference/2.6.0/Dockerfile.neuronx)              | torch-neuronx, transformers-neuronx, neuronx_distributed          | 2.23.0     | inference | trn1/trn2/inf2 | py310  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-neuronx:2.6.0-neuronx-py310-sdk2.23.0-ubuntu22.04`     |
| [PyTorch 2.6.0](https://github.com/aws-neuron/deep-learning-containers/blob/2.23.0/docker/pytorch/training/2.6.0/Dockerfile.neuronx)               | torch-neuronx, transformers-neuronx, neuronx_distributed          | 2.23.0     | training  | trn1/trn2/inf2 | py310  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuronx:2.6.0-neuronx-py310-sdk2.23.0-ubuntu22.04`      |
| [PyTorch 2.5.1](https://github.com/aws-neuron/deep-learning-containers/blob/2.22.0/docker/pytorch/inference/2.5.1/Dockerfile.neuronx)              | torch-neuronx, transformers-neuronx, neuronx_distributed          | 2.22.0     | inference | trn1/trn2/inf2 | py310  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-neuronx:2.5.1-neuronx-py310-sdk2.22.0-ubuntu22.04`     |
| [PyTorch 2.5.1](https://github.com/aws-neuron/deep-learning-containers/blob/2.22.0/docker/pytorch/training/2.5.1/Dockerfile.neuronx)               | torch-neuronx, transformers-neuronx, neuronx_distributed          | 2.22.0     | training  | trn1/trn2/inf2 | py310  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuronx:2.5.1-neuronx-py310-sdk2.22.0-ubuntu22.04`      |
| [PyTorch 2.1.2](https://github.com/aws-neuron/deep-learning-containers/blob/2.20.2/docker/pytorch/inference/2.1.2/Dockerfile.neuronx)              | torch-neuronx, transformers-neuronx, neuronx_distributed          | 2.20.2     | inference | trn1/inf2      | py310  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-neuronx:2.1.2-neuronx-py310-sdk2.20.2-ubuntu20.04`     |
| [PyTorch 2.1.2](https://github.com/aws-neuron/deep-learning-containers/blob/2.20.2/docker/pytorch/training/2.1.2/Dockerfile.neuronx)               | torch-neuronx, neuronx_distributed                                | 2.20.2     | training  | trn1/inf2      | py310  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuronx:2.1.2-neuronx-py310-sdk2.20.2-ubuntu20.04`      |
| [PyTorch 1.13.1](https://github.com/aws-neuron/deep-learning-containers/blob/2.20.2/docker/pytorch/inference/1.13.1/Dockerfile.neuron)             | torch-neuron                                                      | 2.20.2     | inference | inf1           | py310  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-neuron:1.13.1-neuron-py310-sdk2.20.2-ubuntu20.04`      |
| [PyTorch 1.13.1](https://github.com/aws-neuron/deep-learning-containers/blob/2.20.2/docker/pytorch/inference/1.13.1/Dockerfile.neuronx)            | torch-neuronx, transformers-neuronx, neuronx_distributed          | 2.20.2     | inference | trn1/inf2      | py310  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-neuronx:1.13.1-neuronx-py310-sdk2.20.2-ubuntu20.04`    |
| [PyTorch 1.13.1](https://github.com/aws-neuron/deep-learning-containers/blob/2.20.2/docker/pytorch/training/1.13.1/Dockerfile.neuronx)             | torch-neuronx, neuronx_distributed                                | 2.20.2     | training  | trn1/inf2      | py310  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuronx:1.13.1-neuronx-py310-sdk2.20.2-ubuntu20.04`     |
| [TensorFlow 2.10.1](https://github.com/aws/deep-learning-containers/blob/master/tensorflow/inference/docker/2.10/py3/sdk2.17.0/Dockerfile.neuron)  | tensorflow-neuron                                                 | 2.17.0     | inference | inf1           | py310  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference-neuron:2.10.1-neuron-py310-sdk2.17.0-ubuntu20.04`   |
| [TensorFlow 2.10.1](https://github.com/aws/deep-learning-containers/blob/master/tensorflow/inference/docker/2.10/py3/sdk2.17.0/Dockerfile.neuronx) | tensorflow-neuronx                                                | 2.17.0     | inference | trn1/inf2      | py310  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference-neuronx:2.10.1-neuronx-py310-sdk2.17.0-ubuntu20.04` |

---

## Prior Versions

### Prior Neuron Containers

| Framework                                                                                                                                        | Neuron Package    | Neuron SDK | Job Type  | Instance Types | Python | Example URL                                                                                                        |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------- | ---------- | --------- | -------------- | ------ | ------------------------------------------------------------------------------------------------------------------ |
| [TensorFlow 1.15.5](https://github.com/aws/deep-learning-containers/blob/master/tensorflow/inference/docker/1.15/py3/sdk2.8.0/Dockerfile.neuron) | tensorflow-neuron | 2.8.0      | inference | inf1           | py38   | `763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference-neuron:1.15.5-neuron-py38-sdk2.8.0-ubuntu20.04` |
| [MXNet 1.8.0](https://github.com/aws/deep-learning-containers/blob/master/mxnet/inference/docker/1.8/py3/sdk2.5.0/Dockerfile.neuron)             | mx_neuron         | 2.5.0      | inference | inf1           | py38   | `763104351884.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference-neuron:1.8.0-neuron-py38-sdk2.5.0-ubuntu20.04`       |

### Prior EC2 Framework Containers

| Framework     | Job Type | CPU/GPU | Python | Example URL                                                                                           |
| ------------- | -------- | ------- | ------ | ----------------------------------------------------------------------------------------------------- |
| PyTorch 2.8.0 | training | CPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.8.0-cpu-py312-ubuntu22.04-ec2`       |
| PyTorch 2.8.0 | training | GPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2` |
| PyTorch 2.7.1 | training | CPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.7.1-cpu-py312-ubuntu22.04-ec2`       |
| PyTorch 2.7.1 | training | GPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.7.1-gpu-py312-cu128-ubuntu22.04-ec2` |
| PyTorch 2.6.0 | training | CPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.6.0-cpu-py312-ubuntu22.04-ec2`       |
| PyTorch 2.6.0 | training | GPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.6.0-gpu-py312-cu126-ubuntu22.04-ec2` |

### Prior SageMaker Framework Containers

| Framework     | Job Type | CPU/GPU | Python | Example URL                                                                                                 |
| ------------- | -------- | ------- | ------ | ----------------------------------------------------------------------------------------------------------- |
| PyTorch 2.8.0 | training | CPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.8.0-cpu-py312-ubuntu22.04-sagemaker`       |
| PyTorch 2.8.0 | training | GPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker` |
| PyTorch 2.7.1 | training | CPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.7.1-cpu-py312-ubuntu22.04-sagemaker`       |
| PyTorch 2.7.1 | training | GPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.7.1-gpu-py312-cu128-ubuntu22.04-sagemaker` |
| PyTorch 2.6.0 | training | CPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.6.0-cpu-py312-ubuntu22.04-sagemaker`       |
| PyTorch 2.6.0 | training | GPU     | py312  | `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.6.0-gpu-py312-cu126-ubuntu22.04-sagemaker` |

### Prior AutoGluon Containers

#### Training

| Framework       | CPU/GPU | Python | Example URL                                                                                         |
| --------------- | ------- | ------ | --------------------------------------------------------------------------------------------------- |
| AutoGluon 1.3.0 | GPU     | py311  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/autogluon-training:1.3.0-gpu-py311-cu124-ubuntu22.04` |
| AutoGluon 1.3.0 | CPU     | py311  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/autogluon-training:1.3.0-cpu-py311-ubuntu22.04`       |

#### Inference

| Framework       | CPU/GPU | Python | Example URL                                                                                          |
| --------------- | ------- | ------ | ---------------------------------------------------------------------------------------------------- |
| AutoGluon 1.3.0 | GPU     | py311  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/autogluon-inference:1.3.0-gpu-py311-cu124-ubuntu22.04` |
| AutoGluon 1.3.0 | CPU     | py311  | `763104351884.dkr.ecr.us-west-2.amazonaws.com/autogluon-inference:1.3.0-cpu-py311-ubuntu22.04`       |

### Prior SageMaker Training Compiler Containers

| Framework                                                  | CPU/GPU | Python | Example URL                                                                                                                               |
| ---------------------------------------------------------- | ------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| TensorFlow 2.6.3 + transformers 4.17.0 + Training Compiler | GPU     | py38   | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-tensorflow-trcomp-training:2.6.3-transformers4.17.0-gpu-py38-cu112-ubuntu20.04` |
| PyTorch 1.12.0 + Training Compiler                         | GPU     | py38   | `763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-trcomp-training:1.12.0-gpu-py38-cu113-ubuntu20.04-sagemaker`                        |

### Prior HuggingFace Training Containers

| Framework                     | CPU/GPU | Python | Example URL                                                                                                                      |
| ----------------------------- | ------- | ------ | -------------------------------------------------------------------------------------------------------------------------------- |
| PyTorch 1.10.2 + transformers | GPU     | py38   | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04` |

---

## Additional Resources

- [GitHub Releases](https://github.com/aws/deep-learning-containers/releases)
- [ECR Public Gallery](https://gallery.ecr.aws/deep-learning-containers/)
- [Support Policy](support_policy.md)
- [Framework Support Policy](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/support-policy.html)
