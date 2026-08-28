# Amazon SageMaker AI Deployment

Use the TensorFlow Serving DLC to host SavedModel artifacts on {{ sagemaker }} real-time endpoints and batch transform jobs. The images bundle
`tensorflow_model_server`, nginx, and the SageMaker TensorFlow Serving handler stack, so no custom container code is required for the common case.

The TensorFlow Serving 2.20 DLC is a **SageMaker-only** release — there is no EC2 or EKS variant of this image.

## Packaging Model Artifacts

{{ sm_short }} extracts your `model.tar.gz` into `/opt/ml/model`. The container discovers SavedModels by looking for `saved_model.pb` under a numeric
version directory, so the archive must keep the standard TensorFlow Serving layout:

```text
model.tar.gz
└── model/               # model name as served (any name)
    └── 1/               # numeric version directory
        ├── saved_model.pb
        └── variables/
            ├── variables.data-00000-of-00001
            └── variables.index
```

To add pre/post-processing or extra Python dependencies, include a `code/` directory at the root of the archive:

```text
model.tar.gz
├── model/
│   └── 1/
│       └── ...
└── code/
    ├── inference.py
    └── requirements.txt
```

At container start, `requirements.txt` is installed with `pip` before the model server accepts traffic, and `inference.py` is imported by the Python
handler. Multiple version directories under one model name are all loaded and can be addressed individually per request.

## SageMaker Python SDK v2

Pass the DLC image URI via `image_uri=` rather than `framework_version=`:

```python
from sagemaker.tensorflow import TensorFlowModel

model = TensorFlowModel(
    image_uri="public.ecr.aws/deep-learning-containers/tensorflow-inference:2.20-gpu-py312-cu129-amzn2023-sagemaker",
    model_data="s3://<bucket>/models/model.tar.gz",
    role="arn:aws:iam::<account_id>:role/<role_name>",
)

predictor = model.deploy(
    instance_type="ml.g5.xlarge",
    initial_instance_count=1,
)

# The default handler accepts the TensorFlow Serving REST predict payload.
response = predictor.predict({"instances": [[1.0, 2.0, 5.0]]})
print(response)  # {"predictions": [[...]]}

# Cleanup
predictor.delete_model()
predictor.delete_endpoint(delete_endpoint_config=True)
```

For CPU instances (e.g. `ml.c6i`, `ml.m6i`), use the `2.20-cpu-py312-amzn2023-sagemaker` tag instead.

## SageMaker Python SDK v3

```python
import json

import boto3
from sagemaker.core.resources import Endpoint, EndpointConfig, Model
from sagemaker.core.shapes import ContainerDefinition, ProductionVariant

# Requires a configured AWS region (always set inside SageMaker environments).
region = boto3.session.Session().region_name

model = Model.create(
    model_name="tfs-model",
    primary_container=ContainerDefinition(
        image="public.ecr.aws/deep-learning-containers/tensorflow-inference:2.20-gpu-py312-cu129-amzn2023-sagemaker",
        model_data_url="s3://<bucket>/models/model.tar.gz",
    ),
    execution_role_arn="arn:aws:iam::<account_id>:role/<role_name>",
    region=region,
)

ep_cfg = EndpointConfig.create(
    endpoint_config_name="tfs-config",
    production_variants=[
        ProductionVariant(
            variant_name="default",
            model_name="tfs-model",
            instance_type="ml.g5.xlarge",
            initial_instance_count=1,
        ),
    ],
    region=region,
)

endpoint = Endpoint.create(
    endpoint_name="tfs-endpoint",
    endpoint_config_name="tfs-config",
    region=region,
)
endpoint.wait_for_status("InService")

smrt = boto3.client("sagemaker-runtime", region_name=region)
resp = smrt.invoke_endpoint(
    EndpointName="tfs-endpoint",
    ContentType="application/json",
    Body=json.dumps({"instances": [[1.0, 2.0, 5.0]]}),
)
print(json.loads(resp["Body"].read()))  # {"predictions": [[...]]}

# Cleanup
endpoint.delete()
ep_cfg.delete()
model.delete()
```

## Request Formats

All traffic reaches the container as `POST /invocations` on port 8080. When no `inference.py` is supplied, the request is forwarded to the TensorFlow
Serving REST API by nginx. A body that is already a `predict` request (`instances` for row format, `inputs` for columnar format) is passed through as
is; a bare payload is wrapped into an `instances` request first, as shown below.

The default handler accepts these content types:

| `Content-Type` | Notes |
| --- | --- |
| `application/json` | TensorFlow Serving `predict` request body, or a bare payload that is wrapped into an `instances` request |
| `application/jsons` | Concatenated JSON objects, combined into an `instances` request |
| `application/jsonlines` | One JSON object per line, combined into an `instances` request |
| `text/csv` | Rows converted to an `instances` request |

Any other content type is rejected unless your `inference.py` declares an `input_handler` that can parse it.

### Targeting a Specific Model or Version

Pass the `X-Amzn-SageMaker-Custom-Attributes` header to select the model name, version, or TensorFlow Serving method for a single request:

```python
resp = smrt.invoke_endpoint(
    EndpointName="tfs-endpoint",
    ContentType="application/json",
    CustomAttributes="tfs-model-name=model,tfs-model-version=1,tfs-method=predict",
    Body=json.dumps({"instances": [[1.0, 2.0, 5.0]]}),
)
```

## Custom Pre/Post-Processing

Provide `code/inference.py` with either a single `handler(data, context)` function, or an `input_handler(data, context)` /
`output_handler(data, context)` pair. Defining both `handler` and the pair is an error.

```python
import json


def input_handler(data, context):
    """Transform the request payload into a TensorFlow Serving predict body."""
    if context.request_content_type == "application/json":
        payload = json.loads(data.read().decode("utf-8"))
        return json.dumps({"instances": payload["rows"]})
    raise ValueError(f"unsupported content type: {context.request_content_type}")


def output_handler(data, context):
    """Post-process the TensorFlow Serving response."""
    if data.status_code != 200:
        raise ValueError(data.content.decode("utf-8"))
    predictions = json.loads(data.content.decode("utf-8"))["predictions"]
    return json.dumps({"labels": [int(p[0] > 0.5) for p in predictions]}), "application/json"
```

The `context` object exposes the request metadata (`request_content_type`, `accept_header`, `custom_attributes`, `content_length`, the resolved
`model_name` / `model_version` / `method`) along with `rest_uri`, `grpc_port`, and `timeout` for calling TensorFlow Serving directly.

## Multi-Model Endpoints

Set `SAGEMAKER_MULTI_MODEL=true` to host many models behind one endpoint and manage them at runtime through the {{ sm_short }} model-management API.
Models are loaded on demand into `/opt/ml/models/<model_name>/model`, and each may carry its own `code/inference.py`; a universal handler is used for
models that do not.

```python
model = Model.create(
    model_name="tfs-mme",
    primary_container=ContainerDefinition(
        image="public.ecr.aws/deep-learning-containers/tensorflow-inference:2.20-cpu-py312-amzn2023-sagemaker",
        mode="MultiModel",
        model_data_url="s3://<bucket>/models/",
        environment={"SAGEMAKER_MULTI_MODEL": "true"},
    ),
    execution_role_arn="arn:aws:iam::<account_id>:role/<role_name>",
    region=region,
)
```

## Configuration

The container is tuned through environment variables passed in the model's `Environment` map:

| Environment Variable | Purpose |
| --- | --- |
| `SAGEMAKER_TFS_DEFAULT_MODEL_NAME` | Model name to serve when the request does not name one (otherwise derived from the artifact layout) |
| `SAGEMAKER_TFS_ENABLE_BATCHING` | Enable TensorFlow Serving server-side batching (`true`/`false`) |
| `SAGEMAKER_TFS_MAX_BATCH_SIZE` | Maximum number of requests per batch |
| `SAGEMAKER_TFS_BATCH_TIMEOUT_MICROS` | How long to wait for a batch to fill, in microseconds |
| `SAGEMAKER_TFS_NUM_BATCH_THREADS` | Number of threads processing batches |
| `SAGEMAKER_TFS_MAX_ENQUEUED_BATCHES` | Queue depth before requests are rejected |
| `SAGEMAKER_TFS_INTRA_OP_PARALLELISM` | TensorFlow intra-op thread pool size |
| `SAGEMAKER_TFS_INTER_OP_PARALLELISM` | TensorFlow inter-op thread pool size |
| `SAGEMAKER_TFS_INSTANCE_COUNT` | Number of `tensorflow_model_server` processes to run |
| `SAGEMAKER_TFS_WAIT_TIME_SECONDS` | How long to wait for TensorFlow Serving to become ready before giving up |
| `SAGEMAKER_TFS_FRACTIONAL_GPU_MEM_MARGIN` | GPU memory margin when running several model server processes on one GPU |
| `SAGEMAKER_GUNICORN_WORKERS` | Worker count for the Python handler |
| `SAGEMAKER_GUNICORN_THREADS` | Threads per handler worker |
| `SAGEMAKER_GUNICORN_WORKER_CLASS` | Worker class for the Python handler (defaults to `gevent`) |
| `SAGEMAKER_GUNICORN_TIMEOUT_SECONDS` | Handler worker timeout |
| `SAGEMAKER_NGINX_PROXY_READ_TIMEOUT_SECONDS` | nginx read timeout for upstream responses |
| `SAGEMAKER_TFS_NGINX_LOGLEVEL` | nginx error log level |
| `SAGEMAKER_GUNICORN_LOGLEVEL` | Handler log level |
| `SAGEMAKER_MULTI_MODEL` | Enable multi-model endpoint mode |
| `SAGEMAKER_MULTI_MODEL_UNIVERSAL_BUCKET` | S3 bucket to download `code/` from when it is not part of the model artifact — set together with the prefix below |
| `SAGEMAKER_MULTI_MODEL_UNIVERSAL_PREFIX` | S3 prefix for the above; setting both also enables the Python handler |
| `SAGEMAKER_BIND_TO_PORT` | HTTP port to bind, set by {{ sm_short }} when it does not use 8080 |
| `SAGEMAKER_BATCH` | Set by {{ sm_short }} for batch transform jobs |
| `OMP_NUM_THREADS` | OpenMP thread count for the model server |

Batching is off by default. Turning it on trades latency for throughput and is most useful on GPU instances with steady traffic.

## Container Layout

| Path | Purpose |
| --- | --- |
| `/opt/ml/model/` | Extracted `model.tar.gz` — {{ sm_short }} mounts your SavedModel here |
| `/opt/ml/model/code/` | Optional `inference.py` and `requirements.txt` shipped inside the archive |
| `/opt/ml/models/` | Per-model mount points in multi-model endpoint mode |
| `/sagemaker/` | Handler stack, nginx config template, and the `serve` entry point |
| `/opt/venv/` | Python venv with the handler stack and AWS libraries |

## Notes

- The image serves {{ sm_short }} traffic on port 8080 via nginx. TensorFlow Serving's own gRPC and REST ports are container-local and are not
  reachable through a {{ sm_short }} endpoint. They are also assigned at container start — when {{ sm_short }} supplies `SAGEMAKER_SAFE_PORT_RANGE`,
  one non-overlapping gRPC/REST pair per `SAGEMAKER_TFS_INSTANCE_COUNT` is allocated from that range; otherwise the defaults gRPC 9000 and REST 8501
  are used. Read `context.grpc_port` and `context.rest_uri` in your handler instead of hardcoding a port.
- If your `inference.py` needs the `tensorflow` Python package (e.g. to build `tf.Example` protos), add it to `code/requirements.txt`. The framework
  wheel is deliberately not preinstalled; only the TensorFlow Serving gRPC stubs are (`tensorflow-serving-api-gpu` in the GPU image,
  `tensorflow-serving-api` in the CPU image).
- For a baseline driver/AMI compatible with these CUDA 12.9 images, use a current {{ sm_short }} inference instance — the image is labelled
  `com.amazonaws.sagemaker.inference.cuda.verified_versions=12.9` and the entrypoint applies CUDA forward compatibility automatically when the host
  driver is older.
