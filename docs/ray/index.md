# Ray Serve Inference

Pre-built Docker images for deploying ML models with [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) on AWS. Available in CPU and GPU
variants, built on Amazon Linux 2023 with Python 3.13.

## Latest Announcements

*No announcements at this time.*

## Pull Commands

=== "Default"

````
```bash
docker pull {{ images.latest_ray_default_gpu }}
docker pull {{ images.latest_ray_default_cpu }}
```
````

=== "SageMaker"

````
```bash
docker pull {{ images.latest_ray_sagemaker_gpu }}
docker pull {{ images.latest_ray_sagemaker_cpu }}
```
````

Default images are tested on EC2 instances. See [Available Images](../reference/available_images.md) for all image URIs and
[Getting Started](../get_started/index.md) for authentication instructions.

## Packages

For package versions included in each release, see the [Release Notes](../releasenotes/ray/index.md).

## Versioning Strategy

Image tags follow the format `ray:serve-ml-[<platform>-]{cpu|cuda}-v<MAJOR>.<MINOR>.<PATCH>`. The `<platform>` segment is omitted for default images
and present for platform-specific images (e.g. `sagemaker`).

Version bumps follow these rules:

- **MAJOR** — CUDA, Python, or Ray minor version bump or backwards-incompatible changes.
- **MINOR** — CUDA, Python, or Ray patch version bump and backwards-compatible dependency updates or bug fixes.
- **PATCH** — Security patches and backwards-compatible bug fixes that do not change dependency versions.

## Support Policy

| DLC Version | Ray | Python | CUDA | GA Date | End of Patch |
| --- | --- | --- | --- | --- | --- |
| v1.0.0 | 2.54.0 | 3.13 | 12.9.1 | 2026-02-18 | 2027-02-18 |

See [Support Policy](../reference/support_policy.md) for the full lifecycle policy.

## Deployment Guide

### Model Package Structure

Package your model as a `model.tar.gz` with the following layout:

```
model.tar.gz/
├── config.yaml              # Ray Serve application config
├── deployment.py            # Your @serve.deployment class
└── code/
    └── requirements.txt     # Runtime dependencies (optional, installed at startup)
```

Model weights can optionally be placed at the tarball root alongside `config.yaml` and `deployment.py` (extracted to `/opt/ml/model/` at runtime) if
your model doesn't download them at startup.

The `config.yaml` references your deployment module:

```yaml
applications:
  - name: my-app
    route_prefix: /
    import_path: deployment:app
    deployments:
      - name: MyDeployment
        ray_actor_options:
          num_gpus: 1
```

Set `num_gpus` to the number of GPUs allocated per replica (`0` for CPU-only deployments).

The `import_path` follows the format `module:variable` — `deployment` refers to `deployment.py` in the model package, and `app` is the bound
deployment defined at the bottom of that file:

```python
# deployment.py
from ray import serve

@serve.deployment(num_replicas=1)
class MyDeployment:
    def __init__(self):
        # Load model weights, initialize pipeline, etc.
        ...

    async def __call__(self, request):
        # Handle inference request
        ...

app = MyDeployment.bind()
```

### Deployment Paths

The entrypoint resolves the serve target in this priority order:

| Method | Platform | How |
| --- | --- | --- |
| CLI argument | EC2 only | `docker run <image> deployment:app` — overrides `config.yaml` |
| `config.yaml` | EC2 + SageMaker | Auto-detected at `/opt/ml/model/config.yaml` |
| `SM_RAYSERVE_APP` env var | SageMaker only | Fallback when no `config.yaml` is present |

### EC2 Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `RAY_SERVE_HTTP_HOST` | `127.0.0.1` | Set to `0.0.0.0` to expose the endpoint outside the container |
| `RAY_SERVE_HTTP_PORT` | `8000` | HTTP port for Ray Serve |

### Runtime Dependencies

Place a `code/requirements.txt` in your model package. It is installed automatically before the Ray cluster starts. On SageMaker,
[CodeArtifact](https://aws.amazon.com/codeartifact/) is supported via the `CA_REPOSITORY_ARN` environment variable.

## Examples

### EC2 Deployment

Each example below includes the full model package files. The first three download weights automatically on startup. The tabular example requires
pre-trained weights — substitute your own trained model.

#### Sentiment Analysis

Classify text sentiment using [DistilBERT](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english). Model weights download
automatically from HuggingFace on first startup.

Create the model package:

```bash
mkdir -p nlp-model
```

Save `nlp-model/config.yaml`:

```yaml
--8<-- "examples/ray/nlp-model/config.yaml"
```

Save `nlp-model/deployment.py`:

```python
--8<-- "examples/ray/nlp-model/deployment.py"
```

Run the container and send a request:

```bash
docker run -d --gpus all \
  --shm-size=2g \
  -p 8000:8000 \
  -v $(pwd)/nlp-model:/opt/ml/model \
  -e RAY_SERVE_HTTP_HOST=0.0.0.0 \
  {{ images.latest_ray_default_gpu }}

until curl -sf http://localhost:8000/-/healthz > /dev/null; do sleep 5; done

curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely amazing!"}'
```

```json
{
  "predictions": [
    {"label": "POSITIVE", "score": 0.9987}
  ]
}
```

#### Image Classification

Classify images using DenseNet-161 with ImageNet weights from torchvision. Weights download automatically on first startup.

Create the model package:

```bash
mkdir -p cv-model
```

Save `cv-model/config.yaml`:

```yaml
--8<-- "examples/ray/cv-model/config.yaml"
```

!!! note The `autoscaling_config` in `deployment.py` sets `max_replicas: 2`. Each replica requests 1 GPU, so this configuration requires a multi-GPU
instance. On single-GPU instances, reduce `max_replicas` to 1.

Save `cv-model/deployment.py`:

```python
--8<-- "examples/ray/cv-model/deployment.py"
```

Run the container and classify an image:

```bash
docker run -d --gpus all \
  --shm-size=2g \
  -p 8000:8000 \
  -v $(pwd)/cv-model:/opt/ml/model \
  -e RAY_SERVE_HTTP_HOST=0.0.0.0 \
  {{ images.latest_ray_default_gpu }}

# Download test image while container starts up
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg

until curl -sf http://localhost:8000/-/healthz > /dev/null; do sleep 5; done

curl -X POST http://localhost:8000/ \
  -H "Content-Type: image/jpeg" \
  --data-binary @kitten.jpg
```

```json
{
  "predictions": [
    {"class_id": 281, "class_name": "tabby", "probability": 0.5312},
    {"class_id": 282, "class_name": "tiger_cat", "probability": 0.2198},
    {"class_id": 285, "class_name": "Egyptian_cat", "probability": 0.1065},
    {"class_id": 287, "class_name": "lynx", "probability": 0.0742},
    {"class_id": 283, "class_name": "Persian_cat", "probability": 0.0391}
  ]
}
```

#### Audio Transcription

Transcribe speech using [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h) with FFmpeg backend. Weights download automatically on first
startup.

Create the model package:

```bash
mkdir -p audio-model
```

Save `audio-model/config.yaml`:

```yaml
--8<-- "examples/ray/audio-model/config.yaml"
```

Save `audio-model/deployment.py`:

```python
--8<-- "examples/ray/audio-model/deployment.py"
```

Run the container and transcribe audio:

```bash
docker run -d --gpus all \
  --shm-size=2g \
  -p 8000:8000 \
  -v $(pwd)/audio-model:/opt/ml/model \
  -e RAY_SERVE_HTTP_HOST=0.0.0.0 \
  {{ images.latest_ray_default_gpu }}

until curl -sf http://localhost:8000/-/healthz > /dev/null; do sleep 5; done

curl -X POST http://localhost:8000/ \
  -H "Content-Type: audio/wav" \
  --data-binary @audio.wav
```

```json
{
  "transcription": "<transcription depends on audio input>"
}
```

#### Tabular Classification

Classify Iris species from feature vectors using a small PyTorch neural network. This example requires pre-trained weights (`iris_model.pth` and
`norm_params.json`) in the model directory — substitute your own trained model.

Create the model package:

```bash
mkdir -p tabular-model
```

Save `tabular-model/config.yaml`:

```yaml
--8<-- "examples/ray/tabular-model/config.yaml"
```

Save `tabular-model/deployment.py`:

```python
--8<-- "examples/ray/tabular-model/deployment.py"
```

Run the container (CPU — no GPU needed for tabular):

```bash
docker run -d \
  --shm-size=2g \
  -p 8000:8000 \
  -v $(pwd)/tabular-model:/opt/ml/model \
  -e RAY_SERVE_HTTP_HOST=0.0.0.0 \
  {{ images.latest_ray_default_cpu }}

until curl -sf http://localhost:8000/-/healthz > /dev/null; do sleep 5; done

curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

```json
{
  "prediction": "setosa",
  "confidence": 0.9847,
  "probabilities": {"setosa": 0.9847, "versicolor": 0.0112, "virginica": 0.0041}
}
```

### SageMaker Deployment

Install the SageMaker Python SDK v2 (v3 drops the `Model`, `Predictor`, and `Serializer` APIs used below):

```bash
pip install 'sagemaker>=2,<3'
```

To deploy on SageMaker, package your model directory as a tarball, upload to S3, and deploy using the
[SageMaker Python SDK](https://sagemaker.readthedocs.io/en/v2/). The tarball is automatically downloaded and extracted to `/opt/ml/model/` before the
container starts. The container exposes a SageMaker-compatible adapter on port 8080 with `/ping` (health check) and `/invocations` (inference)
endpoints.

!!! warning SageMaker endpoint deployment takes several minutes and incurs costs for the running instance. Remember to delete endpoints when done.

#### Sentiment Analysis

Package the model directory from the EC2 example, upload to S3, and deploy:

```bash
cd nlp-model
tar czf /tmp/nlp-model.tar.gz .
aws s3 cp /tmp/nlp-model.tar.gz s3://<BUCKET>/models/nlp-sentiment/model.tar.gz
```

```python
--8<-- "examples/ray/sagemaker/deploy_sentiment.py"
```

GPU deploys require `inference_ami_version` — the default SageMaker host AMI has incompatible NVIDIA drivers for CUDA 12.9 images. CPU deploys do not
need this. See [ProductionVariant API reference](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProductionVariant.html) for valid
values.

When done, delete the endpoint to stop incurring costs:

```python
predictor.delete_endpoint()
```

The other EC2 examples (image classification, audio, tabular) deploy the same way — package the model directory as a tarball, upload to S3, and use
the same SDK pattern. Use `IdentitySerializer` for binary inputs (images, audio) and the CPU image (`serve-ml-sagemaker-cpu`) for CPU-only models like
tabular.

### Direct App Import

For models that define a Ray Serve app directly in Python without a `config.yaml`, pass the `module:app` import path directly.

On EC2, pass it as a CLI argument:

```bash
docker run -d --gpus all \
  --shm-size=2g \
  -p 8000:8000 \
  -v /path/to/model:/opt/ml/model \
  -e RAY_SERVE_HTTP_HOST=0.0.0.0 \
  -e RAYSERVE_NUM_GPUS=1 \
  {{ images.latest_ray_default_gpu }} \
  deployment:app
```

On SageMaker, set the `SM_RAYSERVE_APP` environment variable. Package your model directory the same way as the sentiment example (tarball uploaded to
S3), but omit `config.yaml`. The `deployment.py` must be at the tarball root — `SM_RAYSERVE_APP=deployment:app` resolves the module from
`/opt/ml/model/`.

```python
--8<-- "examples/ray/sagemaker/deploy_direct_app.py"
```

Without a `config.yaml`, there is no `ray_actor_options` to set `num_gpus`. Instead, the deployment code reads `RAYSERVE_NUM_GPUS` at import time:

```python
import os
from ray import serve

num_gpus = int(os.environ.get("RAYSERVE_NUM_GPUS", "0"))

@serve.deployment(ray_actor_options={"num_gpus": num_gpus})
class MyDeployment:
    ...
```

## Release Notes

See [Ray Release Notes](../releasenotes/ray/index.md) for version history and changelogs.

## Resources

- [Ray Serve Documentation](https://docs.ray.io/en/latest/serve/index.html)
- [GitHub Repository](https://github.com/aws/deep-learning-containers)
- [Available Images](../reference/available_images.md)
