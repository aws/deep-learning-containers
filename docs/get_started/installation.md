# Installation

Set up your environment to use AWS Deep Learning Containers.

## Prerequisites

- AWS CLI v2 installed and configured
- Docker installed (version 20.10 or later)
- IAM permissions to pull from Amazon ECR

## Authenticate with Amazon ECR

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
```

!!! note "Region-specific authentication"

Replace `us-east-1` with your desired region. See [Available Images](../reference/available_images.md) for region-specific ECR URLs.

## Pull a Container Image

=== "PyTorch Training (GPU)"

```bash
docker pull 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-ec2
```

=== "PyTorch Inference (CPU)"

```bash
docker pull 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-cpu-py312-ubuntu22.04-ec2
```

## Verify Installation

```bash
docker run --rm 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-ec2 python -c "import torch; print(torch.__version__)"
```

## Next Steps

- [Quick Start Guide](quickstart.md)
- [Available Images](../reference/available_images.md)
