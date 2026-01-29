# AWS Deep Learning Base Containers for EC2, ECS, EKS (CUDA 13.0)

[AWS Deep Learning Containers (DLCs)](https://aws.amazon.com/machine-learning/containers/) now support Base images that serve as a foundational layer
to build the machine learning environment on EC2, ECS and EKS, with Ubuntu 22.04.

These Base DLCs package the essential deep learning components and dependencies without being tied to a specific framework implementation, providing
users the flexibility to customize the DLCs with their preferred frameworks.

## Release Notes

- Development Tools: Includes curl, build-essential, cmake, and git
- Python Environment: Python 3.12 with AWS CLI, boto3, and requests pre-installed
- GPU Support: CUDA 13.0.0 with cuda-compat for backward compatibility
- Neural Network Libraries: cuDNN 9.13.0.50 for deep neural network operations
- Distributed Training: NCCL 2.27.7-1 for multi-GPU and multi-node communication
- Network Performance: EFA 1.44.0 for low-latency network communications

## Security Advisory

AWS recommends that customers monitor critical security updates in the [AWS Security Bulletin](https://aws.amazon.com/security/security-bulletins/).

## Python Support

Python 3.12 is supported.

## GPU Instance Type Support

- CUDA 13.0
- cuDNN 9.13.0.50
- NCCL 2.27.7-1

## Example URL

```
763104351884.dkr.ecr.us-east-1.amazonaws.com/base:13.0.0-gpu-py312-cu130-ubuntu22.04-ec2
```

## Build and Test

- Built on: c5.18xlarge
- Tested on: p4d.24xlarge, p4de.24xlarge, p5.48xlarge
- Tested with: [openclip](https://github.com/mlfoundations/open_clip), [nccl-tests](https://github.com/NVIDIA/nccl-tests)

## Known Issues

No known issues so far.
