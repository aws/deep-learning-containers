# AWS Deep Learning Containers for PyTorch Training 2.8 on EC2, ECS, EKS

AWS Deep Learning Containers for EC2, ECS, EKS are now available with PyTorch 2.8.

## Announcements

- Introduced containers for PyTorch 2.8 for Training on EC2, ECS, EKS

- Added Python 3.12 support

- Added PyTorch domain libraries: torchtnt 0.2.4, torchdata 0.11.0, torchaudio 2.8.0, torchvision 0.23.0

- Added CUDA 12.9, Ubuntu 22.04 support

## Core Packages

| Package | Version |
| --- | --- |
| Python | 3.12.10 |
| PyTorch | 2.8.0 |
| TorchVision | 0.23.0 |
| TorchAudio | 2.8.0 |
| TorchData | 0.11.0 |
| TorchTNT | 0.2.4 |
| CUDA | 12.9.1 |
| cuDNN | 9.10.2.21 |
| NCCL | 2.27.3-1 |
| EFA | 1.43.1 |
| Transformer Engine | 2.5 |
| Flash Attention | 2.8.3 |
| GDRCopy | 2.5.1 |

## Security Advisory

AWS recommends that customers monitor critical security updates in the [AWS Security Bulletin](https://aws.amazon.com/security/security-bulletins/).

## Reference

### Docker Image URIs

```

763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2

763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.8-gpu-py312-cu129-ubuntu22.04-ec2-v1

763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.8.0-gpu-py312-ec2

763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.8-gpu-py312-ec2

public.ecr.aws/deep-learning-containers/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2

public.ecr.aws/deep-learning-containers/pytorch-training:2.8-gpu-py312-cu129-ubuntu22.04-ec2-v1

public.ecr.aws/deep-learning-containers/pytorch-training:2.8.0-gpu-py312-ec2

public.ecr.aws/deep-learning-containers/pytorch-training:2.8-gpu-py312-ec2
```

### Quick Links

- [Available Images](../../reference/available_images.md)
- [Support Policy](../../reference/support_policy.md)
- [GitHub Repository](https://github.com/aws/deep-learning-containers)
