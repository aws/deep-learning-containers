# Amazon SageMaker AI Deployment

Package your model directory as a tarball, upload to S3, and deploy. The tarball is extracted to `/opt/ml/model/` at runtime. The container runs Ray
Serve on port 8000 internally and exposes `/ping` (health) and `/invocations` (inference) on **port 8080** via a SageMaker adapter.

## Deploy

```python
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# Package and upload model
# tar czf /tmp/model.tar.gz -C model-dir .
# aws s3 cp /tmp/model.tar.gz s3://<bucket>/models/my-model/model.tar.gz

model = Model(
    image_uri="public.ecr.aws/deep-learning-containers/ray:serve-ml-sagemaker-cuda",
    model_data="s3://<bucket>/models/my-model/model.tar.gz",
    role="arn:aws:iam::<account_id>:role/<role_name>",
    predictor_cls=Predictor,
)

predictor = model.deploy(
    instance_type="ml.g5.xlarge",
    initial_instance_count=1,
    inference_ami_version="al2-ami-sagemaker-inference-gpu-3-1",
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
)

response = predictor.predict({"text": "This product is amazing!"})
print(response)

# Cleanup
predictor.delete_model()
predictor.delete_endpoint(delete_endpoint_config=True)
```

## Specifying the Ray Serve Application

The SageMaker entrypoint resolves the Ray Serve target in this order:

1. **`/opt/ml/model/config.yaml`** — recommended; the same `applications:` config used on EC2
2. **`SM_RAYSERVE_APP` environment variable** — fallback, in `module:app` format

Use `config.yaml` unless your model package doesn't include one. Set the env via `Model(..., env={"SM_RAYSERVE_APP": "deployment:app"})`.

## Environment Variables

| Variable | Description |
| --- | --- |
| `SM_RAYSERVE_APP` | Ray Serve import path in `module:app` format. Used only when `/opt/ml/model/config.yaml` is absent |
| `CA_REPOSITORY_ARN` | AWS CodeArtifact repository ARN for installing private packages from `code/requirements.txt` |
| `RAYSERVE_BACKEND_URL` | Internal URL the SageMaker adapter proxies to. Defaults to `http://127.0.0.1:8000` — only override if you change the Ray Serve port |

## Private Dependencies via CodeArtifact

To install packages from a private [CodeArtifact](https://aws.amazon.com/codeartifact/) repository at deploy time, set `CA_REPOSITORY_ARN` to the
repository ARN. The SageMaker entrypoint fetches an auth token via `boto3` and adds the authenticated index URL to `pip install -r requirements.txt`:

```python
model = Model(
    image_uri="public.ecr.aws/deep-learning-containers/ray:serve-ml-sagemaker-cuda",
    model_data="s3://<bucket>/models/my-model/model.tar.gz",
    role="arn:aws:iam::<account_id>:role/<role_name>",
    env={
        "CA_REPOSITORY_ARN": "arn:aws:codeartifact:us-west-2:<account>:repository/<domain>/<repo>",
    },
)
```

The SageMaker execution role must have `codeartifact:GetAuthorizationToken` and `codeartifact:GetRepositoryEndpoint` permissions on the repository. If
CodeArtifact is configured but the lookup fails, the container exits — there is no silent fallback to PyPI.

## Notes

- GPU deployments require `inference_ami_version` — the default SageMaker host AMI has incompatible NVIDIA drivers.
- Use `serve-ml-sagemaker-cpu` for CPU-only models (no `inference_ami_version` needed).
