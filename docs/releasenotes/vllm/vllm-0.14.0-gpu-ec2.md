# AWS Deep Learning Containers for vLLM 0.14.0 on EC2, ECS, EKS

AWS Deep Learning Containers for EC2, ECS, EKS are now available with vLLM 0.14.0.

## Announcements

- Introduced vLLM 0.14.0 containers for EC2, ECS, EKS

- Added Python 3.12 support

## Core Packages

| Package | Version |
| --- | --- |
| vLLM | 0.14.0 |
| PyTorch | 2.9.1 |
| TorchVision | 0.24.1 |
| TorchAudio | 2.9.1 |
| Transformers | 4.57.6 |
| CUDA | 12.9.1 |
| cuDNN | 9.15.0 |
| NCCL | 2.28.3 |
| EFA | 1.46.0 |

## Security Advisory

AWS recommends that customers monitor critical security updates in the [AWS Security Bulletin](https://aws.amazon.com/security/security-bulletins/).

## Reference

### Docker Image URIs

```

763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm:0.14.0-gpu-py312-cu129-ubuntu22.04-ec2

763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm:0.14-gpu-py312-cu129-ubuntu22.04-ec2-v1

763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm:0.14.0-gpu-py312-ec2

763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm:0.14-gpu-py312-ec2

public.ecr.aws/deep-learning-containers/vllm:0.14.0-gpu-py312-cu129-ubuntu22.04-ec2

public.ecr.aws/deep-learning-containers/vllm:0.14-gpu-py312-cu129-ubuntu22.04-ec2-v1

public.ecr.aws/deep-learning-containers/vllm:0.14.0-gpu-py312-ec2

public.ecr.aws/deep-learning-containers/vllm:0.14-gpu-py312-ec2
```

### Quick Links

- [Available Images](../../reference/available_images.md)
- [Support Policy](../../reference/support_policy.md)
- [GitHub Repository](https://github.com/aws/deep-learning-containers)
