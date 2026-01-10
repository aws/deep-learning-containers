# Quick Start

Run your first training or inference job with AWS Deep Learning Containers.

## Training Example

### PyTorch Training on EC2

1. Pull the training container:

   ```bash
   docker pull 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-ec2
   ```

1. Run a training script:

   ```bash
   docker run --gpus all -v $(pwd):/workspace \
     763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-ec2 \
     python /workspace/train.py
   ```

## Inference Example

1. Pull the inference container:

   ```bash
   docker pull 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-ec2
   ```

1. Start the inference server:

   ```bash
   docker run --gpus all -p 8080:8080 -v $(pwd)/model:/model \
     763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-ec2
   ```

## Next Steps

- [EKS Training](../tutorials/content/training/eks/README.md) - Distributed training on Kubernetes
- [vLLM on SageMaker](../tutorials/content/vllm-samples/sagemaker/README.md) - Deploy LLMs with vLLM
- [Available Images](../reference/available_images.md) - Browse all container options
