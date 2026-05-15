# Amazon SageMaker AI Deployment

Package your model directory as a tarball, upload to S3, and deploy. The tarball is extracted to `/opt/ml/model/` at runtime. The container exposes
`/ping` (health check) and `/invocations` (inference) endpoints on port 8080.

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

## Notes

- GPU deployments require `inference_ami_version` — the default SageMaker host AMI has incompatible NVIDIA drivers.
- Use `serve-ml-sagemaker-cpu` for CPU-only models (no `inference_ami_version` needed).
- The container resolves the serve target from `config.yaml` in the model tarball. Alternatively, set `SM_RAYSERVE_APP=deployment:app` as an
  environment variable.
- [CodeArtifact](https://aws.amazon.com/codeartifact/) is supported for private dependencies via `CA_REPOSITORY_ARN`.
