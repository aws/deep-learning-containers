# vLLM

High-throughput LLM serving engine optimized for GPU inference on AWS.

## Pull Commands

=== "EC2 / EKS / ECS"

    ```bash
    docker pull public.ecr.aws/deep-learning-containers/vllm:0.17-gpu-py312-ec2
    ```

    Full tag: `public.ecr.aws/deep-learning-containers/vllm:0.17.1-gpu-py312-cu129-ubuntu22.04-ec2`

=== "SageMaker"

    ```bash
    docker pull public.ecr.aws/deep-learning-containers/vllm:0.17-gpu-py312
    ```

    Full tag: `public.ecr.aws/deep-learning-containers/vllm:0.17.1-gpu-py312-cu129-ubuntu22.04-sagemaker`

Browse all versions on the [ECR Public Gallery](https://gallery.ecr.aws/deep-learning-containers/vllm).

## Key Packages

| Package | Version |
|---------|---------|
| vLLM | 0.17.1 |
| PyTorch | 2.10.0 |
| CUDA | 12.9 |
| NCCL | 2.27.5 |
| EFA | 1.47.0 |
| Python | 3.12 |
| OS | Ubuntu 22.04 |

## Versioning

<!-- TODO: Fill in versioning strategy for vLLM DLCs -->

Lorem ipsum dolor sit amet, consectetur adipiscing elit.

## Support Policy

For GA and End of Patch dates for all vLLM versions, see the [Support Policy](../reference/support_policy.md).

## Release Notes

See [vLLM Release Notes](../releasenotes/vllm/index.md) for the full changelog.

## Guides

- [Deploy vLLM on SageMaker](../tutorials/vllm-samples/sagemaker/README.md) — Deploy and test a SageMaker endpoint with vLLM DLCs
- [Deploy DeepSeek on EKS](../tutorials/vllm-samples/deepseek/eks/README.md) — Run DeepSeek models on Amazon EKS with vLLM
