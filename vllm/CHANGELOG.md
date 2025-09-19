# Changelog

All notable changes to vLLM Deep Learning Containers will be documented in this file.

## [0.10.2] - 2025-09-18
### Updated
- vllm/vllm-openai version `v0.10.2`, see [release note](https://github.com/vllm-project/vllm/releases/tag/v0.10.2) for details.

### Added
- Introducing vLLM ARM64 support for AWS Graviton (g5g) with NVIDIA T4 GPUs, using XFormers/FlashInfer as attention backend and V0 engine for Turing architecture compatibility - [release tag](https://github.com/aws/deep-learning-containers/releases/tag/v1.1-vllm-arm64-ec2-0.10.2-gpu-py312)

### Sample ECR URI
```
763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm-arm64:0.10.2-gpu-py312-cu129-ubuntu22.04-ec2-v1.1 
763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm:0.10.2-gpu-py312-cu129-ubuntu22.04-ec2-v1.0 
763104351884.dkr.ecr.us-east-1.amazonaws.com/vllm:0.10.2-gpu-py312-cu129-ubuntu22.04-ec2
```

## [0.10.1] - 2025-08-25
### Updated
- vllm/vllm-openai version `v0.10.1.1`, see [release note](https://github.com/vllm-project/vllm/releases/tag/v0.10.1.1) for details.
- EFA installer version `1.43.2`
### Sample ECR URI
```
763104351884.dkr.ecr.us-east-1.amazonaws.com/vllm:0.10-gpu-py312-ec2
763104351884.dkr.ecr.us-east-1.amazonaws.com/vllm:0.10.1-gpu-py312-cu128-ubuntu22.04-ec2
```

## [0.10.0] - 2025-08-04
### Updated
- vllm/vllm-openai version `v0.10.0`, see [release note](https://github.com/vllm-project/vllm/releases/tag/v0.10.0) for details.
- EFA installer version `1.43.1`
### Sample ECR URI
```
763104351884.dkr.ecr.us-east-1.amazonaws.com/vllm:0.10-gpu-py312-ec2
763104351884.dkr.ecr.us-east-1.amazonaws.com/vllm:0.10.0-gpu-py312-cu128-ubuntu22.04-ec2
```

## [0.9.2] - 2025-07-15
### Updated
- vllm/vllm-openai version `v0.9.2`, see [release note](https://github.com/vllm-project/vllm/releases/tag/v0.9.2) for details.
### Sample ECR URI
```
763104351884.dkr.ecr.us-east-1.amazonaws.com/vllm:0.9-gpu-py312-ec2
763104351884.dkr.ecr.us-east-1.amazonaws.com/vllm:0.9.2-gpu-py312-cu128-ubuntu22.04-ec2
```

## [0.9.1] - 2025-06-13
### Updated
- vllm/vllm-openai version `v0.9.1`, see [release note](https://github.com/vllm-project/vllm/releases/tag/v0.9.1) for details.
- EFA installer version `1.42.0`
### Sample ECR URI
```
763104351884.dkr.ecr.us-east-1.amazonaws.com/vllm:0.9-gpu-py312-ec2
763104351884.dkr.ecr.us-east-1.amazonaws.com/vllm:0.9.1-gpu-py312-cu128-ubuntu22.04-ec2
```


## [0.9.0.1] - 2025-06-10
### Updated
- vllm/vllm-openai version `v0.9.0.1`, see [release note](https://github.com/vllm-project/vllm/releases/tag/v0.9.0.1) for details.
- EFA installer version `1.41.0`
### Sample ECR URI
```
763104351884.dkr.ecr.us-east-1.amazonaws.com/vllm:0.9-gpu-py312-ec2
763104351884.dkr.ecr.us-east-1.amazonaws.com/vllm:0.9.0-gpu-py312-cu128-ubuntu22.04-ec2
```

## [0.8.5] - 2025-06-02

### Added
- vllm/vllm-openai version `v0.8.5`, see [release note](https://github.com/vllm-project/vllm/releases/tag/v0.8.5) for details.
- EFA installer version `1.40.0`
### Sample ECR URI
```
763104351884.dkr.ecr.us-east-1.amazonaws.com/vllm:0.8-gpu-py312-ec2
763104351884.dkr.ecr.us-east-1.amazonaws.com/vllm:0.8.5-gpu-py312-cu128-ubuntu22.04-ec2
```