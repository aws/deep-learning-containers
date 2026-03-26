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

Set `num_gpus` to the number of GPUs allocated per replica (`0` for CPU-only deployments).

The `import_path` follows the format `module:variable` — `deployment` refers to `deployment.py` in the model package, and `app` is the bound deployment defined at the bottom of that file:

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

Each example below includes the full model package files so you can copy-paste and run immediately. Create a directory, save the files, mount it at `/opt/ml/model`, and start the container.

#### Sentiment Analysis

Classify text sentiment using [DistilBERT](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english). Model weights download automatically from HuggingFace on first startup.

Create the model package:

```bash
mkdir -p nlp-model
```

Save `nlp-model/config.yaml`:

```yaml
applications:
  - name: distilbert
    route_prefix: /
    import_path: deployment:app
    deployments:
      - name: DistilBERTSentiment
        ray_actor_options:
          num_gpus: 1
```

Save `nlp-model/deployment.py`:

```python
from ray import serve
from transformers import pipeline
import torch


@serve.deployment(num_replicas=1)
class DistilBERTSentiment:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=self.device,
        )

    async def __call__(self, request):
        data = await request.json()
        text = data.get("text", "")
        results = self.classifier([text] if isinstance(text, str) else text)
        return {"predictions": results}


app = DistilBERTSentiment.bind()
```

Run the container and send a request:

```bash
docker run -d --gpus all \
  --shm-size=2g \
  -p 8000:8000 \
  -v $(pwd)/nlp-model:/opt/ml/model \
  -e RAY_SERVE_HTTP_HOST=0.0.0.0 \
  {{ images.latest_ray_ec2_gpu }}

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
applications:
  - name: densenet
    route_prefix: /
    import_path: deployment:app
    deployments:
      - name: DenseNetClassifier
        ray_actor_options:
          num_gpus: 1
```

Save `cv-model/deployment.py`:

```python
import io

from PIL import Image
from ray import serve


@serve.deployment(
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)
class DenseNetClassifier:
    def __init__(self):
        import torch
        import torchvision.models as models
        import torchvision.transforms as transforms

        self.model = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.class_names = models.DenseNet161_Weights.IMAGENET1K_V1.meta["categories"]

    async def __call__(self, request):
        import torch

        image_bytes = await request.body()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        top5_prob, top5_idx = torch.topk(probabilities, 5)
        predictions = [
            {
                "class_id": int(top5_idx[i]),
                "class_name": self.class_names[int(top5_idx[i])],
                "probability": float(top5_prob[i]),
            }
            for i in range(5)
        ]
        return {"predictions": predictions}


app = DenseNetClassifier.bind()
```

Run the container and classify an image:

```bash
docker run -d --gpus all \
  --shm-size=2g \
  -p 8000:8000 \
  -v $(pwd)/cv-model:/opt/ml/model \
  -e RAY_SERVE_HTTP_HOST=0.0.0.0 \
  {{ images.latest_ray_ec2_gpu }}

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

Transcribe speech using [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h) with FFmpeg backend (GPU-accelerated decoding on GPU images). Weights download automatically on first startup.

Create the model package:

```bash
mkdir -p audio-model
```

Save `audio-model/config.yaml`:

```yaml
applications:
  - name: wav2vec2
    route_prefix: /
    import_path: deployment:app
    deployments:
      - name: Wav2Vec2Transcription
        ray_actor_options:
          num_gpus: 1
```

Save `audio-model/deployment.py`:

```python
import base64
import io

from ray import serve


@serve.deployment(num_replicas=1)
class Wav2Vec2Transcription:
    def __init__(self):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    async def __call__(self, request):
        import torch
        import torchaudio

        content_type = request.headers.get("content-type", "")
        if "audio/wav" in content_type:
            audio_bytes = await request.body()
        else:
            data = await request.json()
            audio_bytes = base64.b64decode(data.get("audio", data.get("data")))

        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes), backend="ffmpeg")

        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        inputs = self.processor(
            waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return {"transcription": transcription.strip()}


app = Wav2Vec2Transcription.bind()
```

Run the container and transcribe audio:

```bash
docker run -d --gpus all \
  --shm-size=2g \
  -p 8000:8000 \
  -v $(pwd)/audio-model:/opt/ml/model \
  -e RAY_SERVE_HTTP_HOST=0.0.0.0 \
  {{ images.latest_ray_ec2_gpu }}

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

Classify Iris species from feature vectors using a small PyTorch neural network. This example requires trained model weights — run the training script below first to generate them.

Create the model package:

```bash
mkdir -p tabular-model
```

Save `tabular-model/train.py` and run it to generate weights:

```python
"""Train a small Iris classifier and save weights for Ray Serve deployment."""
import json

import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class IrisModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))


iris = load_iris()
X = torch.tensor(iris.data, dtype=torch.float32)
y = torch.tensor(iris.target, dtype=torch.long)
mean, std = X.mean(0), X.std(0)
X_norm = (X - mean) / std

model = IrisModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(200):
    loss = loss_fn(model(X_norm), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "tabular-model/iris_model.pth")
json.dump(
    {"mean": mean.tolist(), "std": std.tolist(), "class_names": iris.target_names.tolist()},
    open("tabular-model/norm_params.json", "w"),
)
print(f"Training complete. Loss: {loss.item():.4f}")
```

```bash
pip install scikit-learn torch
python tabular-model/train.py
```

Save `tabular-model/config.yaml`:

```yaml
applications:
  - name: iris
    route_prefix: /
    import_path: deployment:app
    deployments:
      - name: IrisClassifier
        ray_actor_options:
          num_gpus: 0
```

Save `tabular-model/deployment.py`:

```python
import json
import os

from ray import serve


@serve.deployment(num_replicas=1)
class IrisClassifier:
    def __init__(self):
        import torch
        import torch.nn as nn

        class IrisModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 16)
                self.fc2 = nn.Linear(16, 8)
                self.fc3 = nn.Linear(8, 3)
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))

        model_dir = "/opt/ml/model"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = IrisModel()
        self.model.load_state_dict(
            torch.load(os.path.join(model_dir, "iris_model.pth"), map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

        with open(os.path.join(model_dir, "norm_params.json")) as f:
            norm = json.load(f)
        self.mean = torch.tensor(norm["mean"]).to(self.device)
        self.std = torch.tensor(norm["std"]).to(self.device)
        self.classes = norm["class_names"]

    async def __call__(self, request):
        import torch

        data = await request.json()
        features = data.get("features", data.get("data"))
        x = torch.tensor([features], dtype=torch.float32).to(self.device)
        x_norm = (x - self.mean) / self.std

        with torch.no_grad():
            probs = torch.softmax(self.model(x_norm), dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()

        return {
            "prediction": self.classes[pred_idx],
            "confidence": float(probs[0][pred_idx]),
            "probabilities": {cls: float(probs[0][i]) for i, cls in enumerate(self.classes)},
        }


app = IrisClassifier.bind()
```

Run the container (CPU — no GPU needed for tabular):

```bash
docker run -d \
  --shm-size=2g \
  -p 8000:8000 \
  -v $(pwd)/tabular-model:/opt/ml/model \
  -e RAY_SERVE_HTTP_HOST=0.0.0.0 \
  {{ images.latest_ray_ec2_cpu }}

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

Deploy a model to a SageMaker real-time endpoint using the [SageMaker Python SDK](https://sagemaker.readthedocs.io/en/v2/). The model tarball is automatically downloaded from S3 and extracted to `/opt/ml/model/` before the container starts. The container runs Ray Serve internally on port 8000 and exposes a SageMaker-compatible adapter on port 8080 with `/ping` (health check) and `/invocations` (inference) endpoints.

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

#### Direct App Import (No config.yaml)

For models that define a Ray Serve app directly in Python without a `config.yaml`, use the `SM_RAYSERVE_APP` environment variable to specify the `module:app` import path:

```python
model = Model(
    image_uri="{{ images.latest_ray_sagemaker_gpu }}",
    role="arn:aws:iam::<ACCOUNT>:role/SageMakerExecutionRole",
    model_data="s3://<BUCKET>/models/mnist/model.tar.gz",
    predictor_cls=Predictor,
    env={"SM_RAYSERVE_APP": "deployment:app", "RAYSERVE_NUM_GPUS": "1"},
)

predictor = model.deploy(
    instance_type="ml.g5.xlarge",
    initial_instance_count=1,
    endpoint_name="ray-serve-mnist",
    wait=True,
)
```

`RAYSERVE_NUM_GPUS` is a custom environment variable read by the deployment code to set GPU allocation per replica at import time. It is only needed when using `SM_RAYSERVE_APP` without a `config.yaml` — when using `config.yaml`, set `num_gpus` directly under `ray_actor_options` instead.

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
