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

## Examples

### EC2 Deployment

Start the container with a model tarball extracted to a local directory and mounted at `/opt/ml/model`. The entrypoint auto-detects `config.yaml` in the model directory, or you can pass an explicit path or `module:app` import as a CLI argument.

```bash
docker run -d --gpus all \
  --shm-size=2g \
  -p 8000:8000 \
  -v /path/to/model:/opt/ml/model \
  -e RAY_SERVE_HTTP_HOST=0.0.0.0 \
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

#### Sentiment Analysis (NLP)

Send text for DistilBERT sentiment classification:

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

#### Image Classification (CV)

Send a JPEG image for DenseNet-121 top-5 classification:

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

Send a WAV file for Wav2Vec2 speech-to-text (uses FFmpeg backend with GPU-accelerated decoding on GPU images):

```bash
curl -X POST http://localhost:8000/ \
  -H "Content-Type: audio/wav" \
  --data-binary @audio.wav
```

```json
{
  "transcription": "HELLO WORLD",
  "audio_backend": "ffmpeg"
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

Deploy a model to a SageMaker real-time endpoint using the [SageMaker Python SDK](https://sagemaker.readthedocs.io/). Package your model as a `model.tar.gz` containing a `config.yaml` and model artifacts, upload to S3, then deploy:

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
# {"transcription": "HELLO WORLD", "audio_backend": "ffmpeg"}
```

#### Tabular Classification

For models using `SM_RAYSERVE_APP` to specify the app import path:

```python
model = Model(
    image_uri="{{ images.latest_ray_sagemaker_cpu }}",
    role="arn:aws:iam::<ACCOUNT>:role/SageMakerExecutionRole",
    model_data="s3://<BUCKET>/models/tabular-iris/model.tar.gz",
    predictor_cls=Predictor,
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
