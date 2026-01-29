# AWS Deep Learning Containers for PyTorch Inference ARM64 2.6 on SageMaker

AWS Deep Learning Containers for SageMaker are now available with PyTorch 2.6.

## Announcements

- Introduced containers for PyTorch 2.6.0 for Inference on SageMaker (ARM64)

- Added Python 3.12 support

- Starting with PyTorch 2.6, we are removing Conda from the DLCs and installing all Python packages from PyPI

## Core Packages

| Package | Version |
| --- | --- |
| Python | 3.12.8 |
| PyTorch | 2.6.0 |
| TorchVision | 0.21.0 |
| TorchAudio | 2.6.0 |
| TorchText | 0.18.0 |
| TorchData | 0.10.1 |
| TorchServe | 0.12.0 |
| OpenMPI | 4.1.7 |

## Security Advisory

AWS recommends that customers monitor critical security updates in the [AWS Security Bulletin](https://aws.amazon.com/security/security-bulletins/).

## Reference

### Docker Image URIs

```

763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-arm64:2.6.0-cpu-py312-ubuntu22.04-sagemaker

763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-arm64:2.6-cpu-py312-ubuntu22.04-sagemaker-v1

763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-arm64:2.6.0-cpu-py312

763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-arm64:2.6-cpu-py312

public.ecr.aws/deep-learning-containers/pytorch-inference-arm64:2.6.0-cpu-py312-ubuntu22.04-sagemaker

public.ecr.aws/deep-learning-containers/pytorch-inference-arm64:2.6-cpu-py312-ubuntu22.04-sagemaker-v1

public.ecr.aws/deep-learning-containers/pytorch-inference-arm64:2.6.0-cpu-py312

public.ecr.aws/deep-learning-containers/pytorch-inference-arm64:2.6-cpu-py312
```

### Quick Links

- [Available Images](../../reference/available_images.md)
- [Support Policy](../../reference/support_policy.md)
- [GitHub Repository](https://github.com/aws/deep-learning-containers)
