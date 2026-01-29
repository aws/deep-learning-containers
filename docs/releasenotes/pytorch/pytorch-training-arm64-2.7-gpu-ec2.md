# AWS Deep Learning Containers for PyTorch Training ARM64 2.7 on EC2, ECS, EKS

AWS Deep Learning Containers for EC2, ECS, EKS are now available with PyTorch 2.7.

## Announcements

- Introduced containers for PyTorch 2.7.0 for Training on EC2, ECS, EKS (ARM64)

- Added Python 3.12 support

## Core Packages

| Package | Version |
| --- | --- |
| Python | 3.12.8 |
| PyTorch | 2.7.0 |
| TorchVision | 0.22.0 |
| TorchAudio | 2.7.0 |
| TorchText | 0.18.0 |
| TorchData | 0.11.0 |
| CUDA | 12.8.0 |
| cuDNN | 9.8.0.87 |
| NCCL | 2.27.5 |
| EFA | 1.42.0 |
| Transformer Engine | 2.0 |
| Flash Attention | 2.7.3 |
| GDRCopy | 2.5 |

## Security Advisory

AWS recommends that customers monitor critical security updates in the [AWS Security Bulletin](https://aws.amazon.com/security/security-bulletins/).

## Reference

### Docker Image URIs

```

763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-arm64:2.7.0-gpu-py312-cu128-ubuntu22.04-ec2

763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-arm64:2.7-gpu-py312-cu128-ubuntu22.04-ec2-v1

763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-arm64:2.7.0-gpu-py312-ec2

763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-arm64:2.7-gpu-py312-ec2

public.ecr.aws/deep-learning-containers/pytorch-training-arm64:2.7.0-gpu-py312-cu128-ubuntu22.04-ec2

public.ecr.aws/deep-learning-containers/pytorch-training-arm64:2.7-gpu-py312-cu128-ubuntu22.04-ec2-v1

public.ecr.aws/deep-learning-containers/pytorch-training-arm64:2.7.0-gpu-py312-ec2

public.ecr.aws/deep-learning-containers/pytorch-training-arm64:2.7-gpu-py312-ec2
```

### Quick Links

- [Available Images](../../reference/available_images.md)
- [Support Policy](../../reference/support_policy.md)
- [GitHub Repository](https://github.com/aws/deep-learning-containers)
