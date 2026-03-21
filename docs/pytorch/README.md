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

## Image Architecture

Multi-stage Dockerfile producing two targets:

- **base** — Root user, all platforms. CUDA runtime, EFA, OpenSSH (port 22),
  NCCL tuning, SageMaker `/opt/ml/` paths, GDRCopy userspace lib.
- **eks** — Non-root (`mluser`, uid 1000). SSH on port 2222. Suitable for
  EKS clusters with Pod Security Standards.

Both targets share the same Python venv at `/opt/venv` with all packages
installed via `uv` from a locked dependency set.

## Tag Scheme

```text
pytorch-al2023:2.10.0-cu128-al2023-YYYYMMDD      # date-stamped base
pytorch-al2023:2.10.0-cu128-al2023                # latest base
pytorch-al2023:2.10.0-cu128-al2023-eks-YYYYMMDD   # date-stamped EKS
pytorch-al2023:2.10.0-cu128-al2023-eks             # latest EKS
pytorch-al2023:latest                              # latest base
```

## Build

```bash
source docker/pytorch/versions.env

# Runtime image (root — EC2, ECS, Batch, SageMaker)
docker build --target runtime \
  -t pytorch-al2023:${TORCH_VERSION}-cu128-al2023 \
  -f docker/pytorch/Dockerfile .

# Runtime-EKS image (non-root)
docker build --target runtime-eks \
  -t pytorch-al2023:${TORCH_VERSION}-cu128-al2023-eks \
  -f docker/pytorch/Dockerfile .
```

## Usage Examples

### EC2

```bash
docker run --gpus all --shm-size=1g -v /data:/data \
  pytorch-al2023:2.10.0-cu128-al2023 \
  torchrun --nproc_per_node=8 /data/train.py
```

### EKS (Kubeflow PyTorchJob)

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
spec:
  pytorchReplicaSpecs:
    Master:
      template:
        spec:
          containers:
            - name: pytorch
              image: pytorch-al2023:2.10.0-cu128-al2023-eks
              resources:
                limits:
                  nvidia.com/gpu: 4
    Worker:
      replicas: 3
      template:
        spec:
          containers:
            - name: pytorch
              image: pytorch-al2023:2.10.0-cu128-al2023-eks
              resources:
                limits:
                  nvidia.com/gpu: 4
```

### SageMaker BYOC

```python
from sagemaker.estimator import Estimator

estimator = Estimator(
    image_uri="<account>.dkr.ecr.<region>.amazonaws.com/pytorch-al2023:2.10.0-cu128-al2023",
    role="<role-arn>",
    instance_count=2,
    instance_type="ml.p4d.24xlarge",
    entry_point="train.py",
)
estimator.fit({"training": "s3://bucket/data"})
```

### Multi-node on EC2 (torchrun)

```bash
# Node 0
torchrun --nnodes=2 --nproc_per_node=8 \
  --node_rank=0 --master_addr=<node0-ip> --master_port=29500 \
  train.py

# Node 1
torchrun --nnodes=2 --nproc_per_node=8 \
  --node_rank=1 --master_addr=<node0-ip> --master_port=29500 \
  train.py
```

## Testing

Tests run on the host against the built Docker image via `pytest --image-uri`.

```bash
# Unit tests (no GPU required)
pytest test/pytorch/unit/ --image-uri <image> -v

# Single GPU
pytest test/pytorch/single_gpu/ --image-uri <image> -v

# Multi GPU (needs 2+ GPUs, uses --gpus all --shm-size=1g)
pytest test/pytorch/multi_gpu/ --image-uri <image> -v

# Multi node (spins up 2 containers on a Docker bridge network)
pytest test/pytorch/multi_node/ --image-uri <image> -v

# Full suite
pytest test/pytorch/ --image-uri <image> -v
```

### Test Coverage

| Tier | Tests | What's validated |
|---|---|---|
| unit | 29 | Imports, versions, filesystem, SSH config |
| single_gpu | 11 | CUDA, AMP, torch.compile, Flash Attention, SDPA, Transformer Engine, training |
| multi_gpu | 3 | DDP, FSDP, DeepSpeed ZeRO-2 |
| multi_node | 2 | NCCL all_reduce, multi-node DDP |

## Security Refresh

To update packages for security patches:

```bash
# Update Python dependencies
cd docker/pytorch
uv lock --upgrade
# Rebuild image — dnf update runs during build

# Update system packages only
docker build --no-cache --target runtime ...
```
