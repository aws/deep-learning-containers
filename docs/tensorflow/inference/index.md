# Model Serving using TensorFlow Serving DLC

Production-ready Docker images for serving TensorFlow SavedModel artifacts with [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) on
{{ sagemaker }}. Available in CPU and GPU variants, built on Amazon Linux 2023 with ongoing security patching.

These images pair the `tensorflow_model_server` binary with an nginx front end and the SageMaker TensorFlow Serving handler stack, so a SavedModel in
S3 becomes an HTTPS endpoint without a custom container. Batching, multi-model endpoints, and optional pre/post-processing hooks are all driven by
environment variables and an optional `inference.py`.

## Images

| Platform | Variant | Image |
| --- | --- | --- |
| {{ sagemaker }} | GPU | `public.ecr.aws/deep-learning-containers/tensorflow-inference:2.21-gpu-py312-cu129-amzn2023-sagemaker` |
| {{ sagemaker }} | CPU | `public.ecr.aws/deep-learning-containers/tensorflow-inference:2.21-cpu-py312-amzn2023-sagemaker` |

All images are also available on the [ECR Public Gallery](https://gallery.ecr.aws/deep-learning-containers/tensorflow-inference). For private ECR
URIs, see [Image Access](../../get_started/index.md).

The `2.21.0-*` fully-qualified tags (e.g. `2.21.0-gpu-py312-cu129-amzn2023-sagemaker`) are also published and pin to the same image digest. The
TensorFlow Serving 2.20 images (`2.20-gpu-py312-cu129-amzn2023-sagemaker`, `2.20-cpu-py312-amzn2023-sagemaker`) remain available.

TensorFlow Serving 2.21 is a **SageMaker-only** release — there is no EC2 or EKS variant of this image.

## What's Included

The GPU image includes:

- **TensorFlow Serving 2.21.0** — the `tensorflow_model_server` binary, built with native NVIDIA Blackwell (`sm_120`) support, plus the matching gRPC
  stubs (`tensorflow-serving-api-gpu` in the GPU image, `tensorflow-serving-api` in the CPU image)
- **CUDA 12.9.1** with **cuDNN 9.24.0.43** (`nvidia-cudnn-cu12`), plus `ptxas` and `libdevice` for the GPU kernels TensorFlow compiles at runtime
- **nginx 1.30.5** with the **njs 1.0.1** module — terminates {{ sm_short }} traffic on port 8080 and routes it to TensorFlow Serving or to the Python
  handler
- **SageMaker TensorFlow Serving handler stack** — `falcon` 3.1.0 and `gunicorn` behind `gevent` workers, serving the optional `inference.py`
  pre/post-processing hooks
- **AWS tooling** — `boto3`, `botocore`, `awscli` (<2), `requests`
- **Lightweight data tooling** for handler scripts — `pandas`, `scikit-learn`, `cloudpickle`, `numpy` 1.26.4
- **Python 3.12** in a venv at `/opt/venv` (`PATH` already set)

The CPU variant is identical apart from the CUDA, cuDNN, `ptxas` and `libdevice` layers, which are not present.

Unlike the TensorFlow training DLC, these images do **not** bundle the `tensorflow` framework wheel, EFA, OpenMPI, or NCCL. Serving is a single-host
workload, and the model server does not need the Python framework to load a SavedModel — this keeps the image substantially smaller.

## Ports

| Port | Purpose |
| --- | --- |
| 8080 | nginx — {{ sm_short }} `POST /invocations` and `GET /ping` |
| 8501 by default | TensorFlow Serving REST API (container-local) |
| 9000 by default | TensorFlow Serving gRPC API (container-local) |

On {{ sagemaker }} only the HTTP port is reachable. The image declares the `com.amazonaws.sagemaker.capabilities.accept-bind-to-port` capability, so
{{ sm_short }} may assign a different HTTP port via `SAGEMAKER_BIND_TO_PORT` — for example when the container runs behind an inference pipeline.

The TensorFlow Serving ports are chosen when the container starts, so do not hardcode them. When {{ sm_short }} supplies a `SAGEMAKER_SAFE_PORT_RANGE`
— for example behind an inference pipeline — non-overlapping gRPC and REST port pairs are allocated from that range, one pair per
`SAGEMAKER_TFS_INSTANCE_COUNT`. Otherwise the defaults gRPC 9000 and REST 8501 are used. An `inference.py` handler should always read
`context.grpc_port` and `context.rest_uri` rather than assuming a fixed port.

## Multi-Model Endpoints

The image declares `com.amazonaws.sagemaker.capabilities.multi-models=true` and can host many models behind one endpoint. Set
`SAGEMAKER_MULTI_MODEL=true` and use the {{ sm_short }} model-management API (`/models`) to load, list, and unload models at runtime. Each model may
ship its own `code/inference.py`, or you can supply one universal handler for all of them.

## CUDA Forward Compatibility

The GPU image entrypoint detects host NVIDIA driver versions older than the bundled `cuda-compat` layer and automatically prepends
`/usr/local/cuda/compat` to `LD_LIBRARY_PATH`. No flag or env var needed — the check runs on every container start. The image is labelled
`com.amazonaws.sagemaker.inference.cuda.verified_versions=12.9`.

## How We Build

These images are curated builds tracking the [TensorFlow Serving](https://github.com/tensorflow/serving) project:

- **Built from the upstream TensorFlow Serving release** — the GPU model server is compiled from the TensorFlow Serving 2.21.0 source with CUDA 12.9.1
  for GPU targets `sm_60` through `sm_100` plus `sm_120` and `compute_120`, because the upstream GPU binary has no Blackwell code. The CPU model
  server is taken from the official `tensorflow/serving:2.21.0-devel` image
- **Reproducible** — Python dependencies pinned via `pyproject.toml` + `uv.lock` for every image variant
- **Security-patched** — continuously maintained with security patches from {{ aws }} on an Amazon Linux 2023 base

## NVIDIA Blackwell (G7, G7e)

The 2.21 GPU image runs natively on NVIDIA Blackwell GPUs such as the RTX PRO 4500 and RTX PRO 6000 in G7 and G7e instances (compute capability 12.0).
On these endpoints, set `InferenceAmiVersion` to `al2023-ami-sagemaker-inference-gpu-4-1` (NVIDIA driver 580) in the endpoint configuration's
production variant. The TensorFlow Serving 2.20 GPU image has no Blackwell code and fails on these instances.

TensorFlow Serving 2.20 was the first TensorFlow inference DLC on Amazon Linux 2023. Prior TensorFlow inference DLCs shipped on Ubuntu.

For deployment walkthroughs, see [{{ sagemaker }} Deployment](deployment/sagemaker.md).
