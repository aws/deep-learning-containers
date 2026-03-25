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
| Ray 2.54 | TBD | TBD |

See [Support Policy](../reference/support_policy.md) for the full lifecycle policy.

## Examples

### Image Classification on EC2

Pull the GPU image, mount a model directory, and send an image for classification:

```bash
# Run the container with a model mounted at /opt/ml/model
docker run -d --gpus all \
  --shm-size=2g \
  -p 8000:8000 \
  -v /path/to/model:/opt/ml/model \
  -e RAY_SERVE_HTTP_HOST=0.0.0.0 \
  {{ images.latest_ray_ec2_gpu }}

# Wait for Ray Serve to become healthy
until curl -s http://localhost:8000/-/healthz | grep -q "OK"; do sleep 5; done

# Send an image for classification
curl -X POST http://localhost:8000/ \
  -H "Content-Type: image/jpeg" \
  --data-binary @image.jpg
```

Response:

```json
{
  "predictions": [
    {"class_id": 281, "class_name": "tabby", "probability": 0.5312},
    {"class_id": 282, "class_name": "tiger_cat", "probability": 0.2198}
  ]
}
```

### Tabular Inference on EC2

Deploy an Iris classification model and send JSON feature vectors:

```bash
docker run -d \
  --shm-size=2g \
  -p 8000:8000 \
  -v /path/to/tabular-model:/opt/ml/model \
  -e RAY_SERVE_HTTP_HOST=0.0.0.0 \
  {{ images.latest_ray_ec2_cpu }} \
  /opt/ml/model/config.yaml

# Wait for health check
until curl -s http://localhost:8000/-/healthz | grep -q "OK"; do sleep 5; done

# Send a prediction request
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

Response:

```json
{
  "prediction": "setosa",
  "confidence": 0.9847
}
```

### SageMaker Endpoint Deployment

Deploy a Ray Serve model to a SageMaker real-time endpoint using the SageMaker Python SDK:

```python
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import IdentitySerializer

image_uri = "{{ images.latest_ray_sagemaker_gpu }}"
model_data = "s3://my-bucket/models/cv-densenet/model.tar.gz"

model = Model(
    image_uri=image_uri,
    role="arn:aws:iam::<account_id>:role/SageMakerExecutionRole",
    model_data=model_data,
    predictor_cls=Predictor,
)

predictor = model.deploy(
    instance_type="ml.g5.xlarge",
    initial_instance_count=1,
    endpoint_name="ray-serve-densenet",
    wait=True,
)

# Send an image for inference
predictor.serializer = IdentitySerializer(content_type="image/jpeg")
with open("image.jpg", "rb") as f:
    response = predictor.predict(f.read())

print(response)

# Clean up
predictor.delete_endpoint()
```

## Release Notes

See [Ray Release Notes](../releasenotes/ray/index.md) for version history and changelogs.

## Resources

- [Ray Serve Documentation](https://docs.ray.io/en/latest/serve/index.html)
- [GitHub Repository](https://github.com/aws/deep-learning-containers)
- [Available Images](../reference/available_images.md)
