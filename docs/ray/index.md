# Ray Serve Inference

Production-ready Docker images for deploying ML models with [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) on {{ aws }}. Available in CPU and GPU variants, built on Amazon Linux 2023 with ongoing security patching.

Ray Serve is a scalable model serving library for deploying any Python model — NLP, computer vision, audio, tabular, and multi-model compositions — behind a single HTTP endpoint.

## Images

| Platform | Variant | Image |
|---|---|---|
| EC2 / EKS | GPU | `public.ecr.aws/deep-learning-containers/ray:serve-ml-cuda` |
| EC2 / EKS | CPU | `public.ecr.aws/deep-learning-containers/ray:serve-ml-cpu` |
| Amazon SageMaker AI | GPU | `public.ecr.aws/deep-learning-containers/ray:serve-ml-sagemaker-cuda` |
| Amazon SageMaker AI | CPU | `public.ecr.aws/deep-learning-containers/ray:serve-ml-sagemaker-cpu` |

For private ECR URIs, see [Image Access](../get_started/index.md).

## How We Build

These images are curated builds tracking the [Ray](https://github.com/ray-project/ray) project:

- **Built from upstream releases** — images track Ray stable releases, each gated by our test suite before publication.
- **Security-patched** — continuously maintained with security patches from {{ aws }} on an Amazon Linux 2023 base.

Each image includes Ray Serve, PyTorch, CUDA (GPU variant), and common ML libraries.
