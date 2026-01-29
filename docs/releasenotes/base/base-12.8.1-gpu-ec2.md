# AWS Deep Learning Containers for Base 12.8.1 on EC2, ECS, EKS

AWS Deep Learning Containers for EC2, ECS, EKS are now available with Base 12.8.1.

## Announcements

- Introduced Base containers with CUDA 12.8.1 for EC2, ECS, EKS

- Added Python 3.12 support

- Ubuntu 24.04 support

## Core Packages

| Package | Version |
| --- | --- |
| Python | 3.12.10 |
| CUDA | 12.8.1 |
| cuDNN | 9.8.0.87 |
| NCCL | 2.26.2-1 |
| EFA | 1.42.0 |

## Security Advisory

AWS recommends that customers monitor critical security updates in the [AWS Security Bulletin](https://aws.amazon.com/security/security-bulletins/).

## Reference

### Docker Image URIs

```

763104351884.dkr.ecr.us-west-2.amazonaws.com/base:12.8.1-gpu-py312-cu128-ubuntu24.04-ec2

763104351884.dkr.ecr.us-west-2.amazonaws.com/base:12.8-gpu-py312-cu128-ubuntu24.04-ec2-v1

763104351884.dkr.ecr.us-west-2.amazonaws.com/base:12.8.1-gpu-py312-ubuntu24.04-ec2

763104351884.dkr.ecr.us-west-2.amazonaws.com/base:12.8-gpu-py312-ubuntu24.04-ec2

public.ecr.aws/deep-learning-containers/base:12.8.1-gpu-py312-cu128-ubuntu24.04-ec2

public.ecr.aws/deep-learning-containers/base:12.8-gpu-py312-cu128-ubuntu24.04-ec2-v1

public.ecr.aws/deep-learning-containers/base:12.8.1-gpu-py312-ubuntu24.04-ec2

public.ecr.aws/deep-learning-containers/base:12.8-gpu-py312-ubuntu24.04-ec2
```

### Quick Links

- [Available Images](../../reference/available_images.md)
- [Support Policy](../../reference/support_policy.md)
- [GitHub Repository](https://github.com/aws/deep-learning-containers)
