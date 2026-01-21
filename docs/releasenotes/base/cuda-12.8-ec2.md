# AWS Deep Learning Base Containers for EC2, ECS, EKS (CUDA 12.8)

[AWS Deep Learning Containers (DLCs)](https://aws.amazon.com/machine-learning/containers/) now support Base images that serve as a foundational layer
to build the machine learning environment on EC2, ECS and EKS, with Ubuntu 24.04.

These Base DLCs package the essential deep learning components and dependencies without being tied to a specific framework implementation, providing
users the flexibility to customize the DLCs with their preferred frameworks.

## Release Notes

- Development Tools: Includes curl, build-essential, cmake, and git
- Python Environment: Python 3.12 with AWS CLI, boto3, and requests pre-installed
- GPU Support: CUDA 12.8.1 with cuda-compat for backward compatibility
- Neural Network Libraries: cuDNN 9.8.0.87 for deep neural network operations
- Distributed Training: NCCL 2.26.2-1 for multi-GPU and multi-node communication
- Network Performance: EFA 1.40.0 for low-latency network communications

## Security Advisory

AWS recommends that customers monitor critical security updates in the [AWS Security Bulletin](https://aws.amazon.com/security/security-bulletins/).

## Python Support

Python 3.12 is supported.

## GPU Instance Type Support

- CUDA 12.8
- cuDNN 9.8.0.87
- NCCL 2.26.2-1

## Example URL

```
763104351884.dkr.ecr.us-east-1.amazonaws.com/base:12.8.1-gpu-py312-cu128-ubuntu24.04-ec2
```

## Build and Test

- Built on: c5.18xlarge
- Tested on: p4d.24xlarge, p4de.24xlarge, p5.48xlarge
- Tested with: [openclip](https://github.com/mlfoundations/open_clip), [nccl-tests](https://github.com/NVIDIA/nccl-tests)

## Known Issues

No known issues so far.
