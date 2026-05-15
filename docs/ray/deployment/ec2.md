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
  - name: my-app
    route_prefix: /
    import_path: deployment:app
    deployments:
      - name: MyDeployment
        ray_actor_options:
          num_gpus: 1
```

The `deployment.py` defines your model:

```python
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

## Run

```bash
docker run -d --gpus all \
  --shm-size=2g \
  -p 8000:8000 \
  -v $(pwd)/model-dir:/opt/ml/model \
  -e RAY_SERVE_HTTP_HOST=0.0.0.0 \
  public.ecr.aws/deep-learning-containers/ray:serve-ml-cuda
```

Wait for readiness and send a request:

```bash
until curl -sf http://localhost:8000/-/healthz > /dev/null; do sleep 5; done

curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely amazing!"}'
```

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `RAY_SERVE_HTTP_HOST` | `127.0.0.1` | Set to `0.0.0.0` to expose outside the container |
| `RAY_SERVE_HTTP_PORT` | `8000` | HTTP port for Ray Serve |

## Runtime Dependencies

Place a `code/requirements.txt` in your model package. Dependencies are installed automatically before the Ray cluster starts.

## Direct App Import

For models that define a Ray Serve app without a `config.yaml`, pass the import path as a CLI argument:

```bash
docker run -d --gpus all \
  --shm-size=2g \
  -p 8000:8000 \
  -v /path/to/model:/opt/ml/model \
  -e RAY_SERVE_HTTP_HOST=0.0.0.0 \
  -e RAYSERVE_NUM_GPUS=1 \
  public.ecr.aws/deep-learning-containers/ray:serve-ml-cuda \
  deployment:app
```
