# EC2 Deployment

## Model Package Structure

Package your model as a directory mounted to `/opt/ml/model/`:

```
model-dir/
├── config.yaml              # Ray Serve application config
├── deployment.py            # Your @serve.deployment class
└── code/
    └── requirements.txt     # Runtime dependencies (optional, installed at startup)
```

The `config.yaml` references your deployment module:

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

GPU allocation is declared per-deployment in `config.yaml` via `ray_actor_options.num_gpus`. The container auto-detects available GPUs via
`nvidia-smi` and starts the Ray cluster with that count.

The `deployment.py` defines your model. This NLP example loads DistilBERT and exposes a sentiment-analysis endpoint:

```python
import torch
from ray import serve
from transformers import pipeline

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

## Run

```bash
docker run -d --gpus all \
  --shm-size=2g \
  -p 8000:8000 \
  -v $(pwd)/model-dir:/opt/ml/model \
  -e RAY_SERVE_HTTP_HOST=0.0.0.0 \
  public.ecr.aws/deep-learning-containers/ray:serve-ml-cuda
```

Wait for readiness and send a request. `/-/healthz` is Ray Serve's built-in health endpoint:

```bash
until curl -sf http://localhost:8000/-/healthz > /dev/null; do sleep 5; done

curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely amazing!"}'
# {"predictions": [{"label": "POSITIVE", "score": 0.9998}]}
```

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `RAY_SERVE_HTTP_HOST` | `127.0.0.1` | Bind address for Ray Serve. Defaults to localhost so the endpoint is not exposed outside the container by accident — set to `0.0.0.0` to publish via `-p` |
| `RAY_SERVE_HTTP_PORT` | `8000` | HTTP port for Ray Serve |

## Runtime Dependencies

Place a `code/requirements.txt` in your model package. The entrypoint runs `pip install -r` before the Ray cluster starts, so additional Python deps
(beyond what's bundled in the image) are available at deploy time. CodeArtifact (private package indices) is not supported on the EC2 image — use the
SageMaker image if you need that.

## Direct App Import

For models that define a Ray Serve app without a `config.yaml`, pass the import path as a CLI argument. The Ray cluster auto-detects GPUs from
`nvidia-smi`; per-deployment GPU allocation belongs in your `@serve.deployment` decorator or a `config.yaml`.

```bash
docker run -d --gpus all \
  --shm-size=2g \
  -p 8000:8000 \
  -v /path/to/model:/opt/ml/model \
  -e RAY_SERVE_HTTP_HOST=0.0.0.0 \
  public.ecr.aws/deep-learning-containers/ray:serve-ml-cuda \
  deployment:app
```

If your `deployment.py` reads its own environment variable for parameterization (e.g., `os.getenv("RAYSERVE_NUM_GPUS")` to set
`ray_actor_options.num_gpus`), pass it via `-e`. The DLC image itself doesn't define such variables — they're a user-side convention specific to your
`deployment.py`.

```bash
docker run -d --gpus all -e RAYSERVE_NUM_GPUS=1 ... deployment:app
```
