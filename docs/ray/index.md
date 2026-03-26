# Ray Serve Inference

Pre-built Docker images for deploying ML models with [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) on AWS. Available in CPU and GPU variants, built on Amazon Linux 2023 with Python 3.13.

## Pull Commands

=== "EC2"

    ```bash
    docker pull {{ images.latest_ray_ec2_gpu }}
    docker pull {{ images.latest_ray_ec2_cpu }}
    ```

=== "SageMaker"

    ```bash
    docker pull {{ images.latest_ray_sagemaker_gpu }}
    docker pull {{ images.latest_ray_sagemaker_cpu }}
    ```

See [Available Images](../reference/available_images.md) for all image URIs and [Getting Started](../get_started/index.md) for authentication instructions.

## Packages

For package versions included in each release, see the [Release Notes](../releasenotes/ray/index.md).

## Versioning Strategy

Image tags follow the format `ray:serve-ml-<platform>-{cpu|cuda}-v<MAJOR>.<MINOR>.<PATCH>`.

Version bumps follow these rules:

- **MAJOR** — CUDA, Python, or Ray minor version bump (e.g. Ray 2.54 → 2.55, Python 3.13 → 3.14, CUDA 12.9 → 13.0).
- **MINOR** — CUDA, Python, or Ray patch version bump and backwards-compatible dependency updates or bug fixes.
- **PATCH** — Security patches and backwards-compatible bug fixes that do not change dependency versions.

## Support Policy

| Version | GA Date | End of Patch |
| ------- | ------- | ------------ |
| Ray 2.54 | 2026-02-18 | 2027-02-18 |

See [Support Policy](../reference/support_policy.md) for the full lifecycle policy.

## Deployment Guide

### Model Package Structure

Package your model as a `model.tar.gz` with the following layout:

```
model.tar.gz/
├── config.yaml              # Ray Serve application config
├── deployment.py            # Your @serve.deployment class
├── model_weights.pth        # Model weights (if applicable)
└── code/
    └── requirements.txt     # Runtime dependencies (optional, installed at startup)
```

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

Set `num_gpus: 1` for GPU images or `num_gpus: 0` for CPU images.

### Deployment Paths

The entrypoint resolves the serve target in this priority order:

| Method | Platform | How |
| ------ | -------- | --- |
| `config.yaml` (default) | EC2 + SageMaker | Place `config.yaml` at root of model package |
| CLI argument | EC2 only | `docker run <image> deployment:app` |
| Environment variable | SageMaker only | Set `SM_RAYSERVE_APP=deployment:app` |

### EC2 Environment Variables

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `RAY_SERVE_HTTP_HOST` | `127.0.0.1` | Set to `0.0.0.0` to expose the endpoint outside the container |
| `RAY_SERVE_HTTP_PORT` | `8000` | HTTP port for Ray Serve |

### Runtime Dependencies

Place a `code/requirements.txt` in your model package. It is installed automatically before the Ray cluster starts. On SageMaker, CodeArtifact is supported via the `CA_REPOSITORY_ARN` environment variable.

## Examples

### EC2 Deployment

Mount a model directory at `/opt/ml/model` and set `RAY_SERVE_HTTP_HOST=0.0.0.0` to expose the endpoint:

```bash
docker run -d --gpus all \
  --shm-size=2g \
  -p 8000:8000 \
  -v /path/to/model:/opt/ml/model \
  -e RAY_SERVE_HTTP_HOST=0.0.0.0 \
  -e RAY_SERVE_HTTP_PORT=8000 \
  {{ images.latest_ray_ec2_gpu }}

# Wait for Ray Serve to become healthy
until curl -s http://localhost:8000/-/healthz | grep -q "OK"; do sleep 5; done
```

For CPU models, use the CPU image and omit `--gpus all`:

```bash
docker run -d \
  --shm-size=2g \
  -p 8000:8000 \
  -v /path/to/model:/opt/ml/model \
  -e RAY_SERVE_HTTP_HOST=0.0.0.0 \
  {{ images.latest_ray_ec2_cpu }}
```

#### Sentiment Analysis

Send text for [DistilBERT](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) sentiment classification:

```bash
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

Send a JPEG image for DenseNet-161 top-5 classification (ImageNet weights via torchvision):

```bash
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

Send a WAV file for [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h) speech-to-text (uses FFmpeg backend with GPU-accelerated decoding on GPU images):

```bash
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

Send feature vectors for Iris species classification. Pass the config path explicitly as a CLI argument:

```bash
docker run -d \
  --shm-size=2g \
  -p 8000:8000 \
  -v /path/to/tabular-model:/opt/ml/model \
  -e RAY_SERVE_HTTP_HOST=0.0.0.0 \
  {{ images.latest_ray_ec2_cpu }} \
  /opt/ml/model/config.yaml
```

```bash
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

Deploy a model to a SageMaker real-time endpoint using the [SageMaker Python SDK](https://sagemaker.readthedocs.io/). The container runs Ray Serve internally on port 8000 and exposes a SageMaker-compatible adapter on port 8080 with `/ping` (health check) and `/invocations` (inference) endpoints.

```python
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer, IdentitySerializer

model = Model(
    image_uri="{{ images.latest_ray_sagemaker_gpu }}",
    role="arn:aws:iam::<ACCOUNT>:role/SageMakerExecutionRole",
    model_data="s3://<BUCKET>/models/nlp-sentiment/model.tar.gz",
    predictor_cls=Predictor,
)

predictor = model.deploy(
    instance_type="ml.g5.xlarge",
    initial_instance_count=1,
    endpoint_name="ray-serve-nlp",
    serializer=JSONSerializer(),
    wait=True,
)
```

#### Sentiment Analysis

```python
response = predictor.predict({"text": "I love this so much, best purchase ever!"})
# {"predictions": [{"label": "POSITIVE", "score": 0.9991}]}
```

#### Image Classification

```python
predictor.serializer = IdentitySerializer(content_type="image/jpeg")
with open("kitten.jpg", "rb") as f:
    response = predictor.predict(f.read())
# {"predictions": [{"class_id": 281, "class_name": "tabby", "probability": 0.5312}, ...]}
```

#### Audio Transcription

```python
predictor.serializer = IdentitySerializer(content_type="audio/wav")
with open("audio.wav", "rb") as f:
    response = predictor.predict(f.read())
# {"transcription": "<transcription depends on audio input>"}
```

#### Tabular Classification

For models using `SM_RAYSERVE_APP` to specify the app import path:

```python
model = Model(
    image_uri="{{ images.latest_ray_sagemaker_cpu }}",
    role="arn:aws:iam::<ACCOUNT>:role/SageMakerExecutionRole",
    model_data="s3://<BUCKET>/models/tabular-iris/model.tar.gz",
    predictor_cls=Predictor,
    env={"SM_RAYSERVE_APP": "deployment:app"},
)

predictor = model.deploy(
    instance_type="ml.m5.xlarge",
    initial_instance_count=1,
    endpoint_name="ray-serve-tabular",
    serializer=JSONSerializer(),
    wait=True,
)

response = predictor.predict({"features": [6.3, 3.3, 6.0, 2.5]})
# {"prediction": "virginica", "confidence": 0.9723, "probabilities": {"setosa": 0.0031, ...}}
```

#### Cleanup

```python
predictor.delete_endpoint()
```

## Release Notes

See [Ray Release Notes](../releasenotes/ray/index.md) for version history and changelogs.

## Resources

- [Ray Serve Documentation](https://docs.ray.io/en/latest/serve/index.html)
- [GitHub Repository](https://github.com/aws/deep-learning-containers)
- [Available Images](../reference/available_images.md)
