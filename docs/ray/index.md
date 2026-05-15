# Ray Serve Inference

Production-ready Docker images for deploying ML models with [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) on {{ aws }}. Available in CPU
and GPU variants, built on Amazon Linux 2023 with ongoing security patching.

Ray Serve is a scalable model serving library for deploying any Python model — NLP, computer vision, audio, tabular, and multi-model compositions —
behind a single HTTP endpoint.

## Images

| Platform | Variant | Image | Default Port |
| --- | --- | --- | --- |
| EC2 / EKS | GPU | `public.ecr.aws/deep-learning-containers/ray:serve-ml-cuda` | 8000 |
| EC2 / EKS | CPU | `public.ecr.aws/deep-learning-containers/ray:serve-ml-cpu` | 8000 |
| Amazon SageMaker AI | GPU | `public.ecr.aws/deep-learning-containers/ray:serve-ml-sagemaker-cuda` | 8080 |
| Amazon SageMaker AI | CPU | `public.ecr.aws/deep-learning-containers/ray:serve-ml-sagemaker-cpu` | 8080 |

For private ECR URIs, see [Image Access](../get_started/index.md).

## What's Included

The images bundle a curated stack so you can ship a serving endpoint without building a custom image:

- **[Ray Serve](https://docs.ray.io/en/latest/serve/index.html) 2.55** — scalable model serving with autoscaling, fractional GPU sharing, and
  multi-model composition
- **[PyTorch](https://pytorch.org/) 2.10** with **CUDA 12.9** (GPU variant) — current stable PyTorch
- **[Transformers](https://huggingface.co/docs/transformers) 5.8** — Hugging Face model loading and `pipeline()` API
- **[FFmpeg](https://ffmpeg.org/) 8.0.1** — built from source for video ingestion and processing pipelines
- **OpenCV, Pillow, soundfile, torchaudio, torchvision, torchcodec** — common image, audio, and video I/O libraries
- **scikit-learn, NumPy, pandas** — for tabular models and feature engineering
- **boto3, awscli** — AWS SDK pre-installed
- **uvicorn[standard], httpx, FastAPI** — async HTTP stack used by Ray Serve and the SageMaker adapter
- **Python 3.13** — built from source with security hardening

## Example Deployments

The repo includes runnable examples for the most common use cases:

| Example | Use case | Path |
| --- | --- | --- |
| **DistilBERT** | NLP / sentiment analysis | [`examples/ray/nlp-model`](https://github.com/aws/deep-learning-containers/tree/main/examples/ray/nlp-model) |
| **DenseNet-161** | Computer vision / image classification | [`examples/ray/cv-model`](https://github.com/aws/deep-learning-containers/tree/main/examples/ray/cv-model) |
| **Wav2Vec2** | Audio / speech-to-text | [`examples/ray/audio-model`](https://github.com/aws/deep-learning-containers/tree/main/examples/ray/audio-model) |
| **Iris classifier** | Tabular / scikit-learn | [`examples/ray/tabular-model`](https://github.com/aws/deep-learning-containers/tree/main/examples/ray/tabular-model) |

## How We Build

These images are curated builds tracking the [Ray](https://github.com/ray-project/ray) project:

- **Built from upstream releases** — images track Ray stable releases, each gated by our test suite before publication.
- **Security-patched** — continuously maintained with security patches from {{ aws }} on an Amazon Linux 2023 base.
