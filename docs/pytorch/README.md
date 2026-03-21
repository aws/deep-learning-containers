# PyTorch 2.10.0 DLC — Amazon Linux 2023

Pre-built Deep Learning Container for PyTorch training on AWS.

## Version Matrix

| Component | Version |
|---|---|
| PyTorch | 2.10.0 |
| CUDA | 12.8.1 |
| cuDNN | 9.8.0 |
| NCCL | 2.26.2 |
| Python | 3.12 |
| Flash Attention | 2.7.4.post1 |
| Transformer Engine | 2.3.0 |
| DeepSpeed | 0.16.7 |
| EFA Installer | 1.47.0 |
| GDRCopy | 2.4.4 |
| OS | Amazon Linux 2023 |

## Supported Platforms

| Platform | Single-node | Multi-node | EFA | Non-root |
|---|---|---|---|---|
| EC2 | ✅ | ✅ | ✅ | ❌ |
| EKS | ✅ | ✅ | ✅ | ✅ |
| ECS | ✅ | Limited | ❌ | ❌ |
| SageMaker BYOC | ✅ | ✅ | ✅ | ❌ |
| AWS Batch | ✅ | ✅ | Optional | ❌ |
| HyperPod | ✅ | ✅ | ✅ | ❌ |

## Build

```bash
source docker/pytorch/versions.env

# Base image (root, all platforms)
docker build --target base \
  -t pytorch-al2023:${TORCH_VERSION}-cu${CUDA_VERSION%%.*}${CUDA_VERSION#*.}-al2023 \
  -f docker/pytorch/Dockerfile .

# EKS image (non-root)
docker build --target eks \
  -t pytorch-al2023:${TORCH_VERSION}-cu${CUDA_VERSION%%.*}${CUDA_VERSION#*.}-al2023-eks \
  -f docker/pytorch/Dockerfile .
```

## Testing

```bash
# Unit tests (no GPU)
pytest test/pytorch/unit/ --image-uri <image> -v

# Single GPU
pytest test/pytorch/single_gpu/ --image-uri <image> -v

# Multi GPU
pytest test/pytorch/multi_gpu/ --image-uri <image> -v

# Multi node
pytest test/pytorch/multi_node/ --image-uri <image> -v
```
