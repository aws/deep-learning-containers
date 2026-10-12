# Lambda Managed Instances Deployment

GPU functions run on [{{ lambda }} Managed Instances](https://docs.aws.amazon.com/lambda/latest/dg/lambda-managed-instances.html) with a GPU capacity
provider. They are container-image functions only — no `.zip` packages, no managed runtimes.

This page covers the image side. For capacity providers, scaling, networking, quotas, and GPU metrics, see the
[{{ lambda_short }} Developer Guide](https://docs.aws.amazon.com/lambda/latest/dg/lambda-managed-instances.html).

## Prerequisites

- A capacity provider whose `AllowedInstanceTypes` lists only GPU instance types. GPU and non-GPU types cannot be mixed.
- A function execution role. Add `s3:GetObject` and `s3:ListBucket` on your model prefix if you load weights from {{ amazon }} S3.
- A route from the capacity provider's subnets to S3 (gateway VPC endpoint or NAT gateway), or to the internet if you pull from the Hugging Face Hub.

## Serve an LLM with no handler code

The vLLM and SGLang images ship a default handler. Set `MODEL_ID` and the image starts the server, loads the model during initialization, and proxies
invocations to it.

```dockerfile
FROM public.ecr.aws/lambda/deep-learning-containers/vllm:cuda-v1

ENV MODEL_ID=s3://amzn-s3-demo-bucket/models/Qwen2.5-1.5B-Instruct/
ENV VLLM_LOAD_FORMAT=runai_streamer
```

`VLLM_LOAD_FORMAT=runai_streamer` streams safetensors shards from S3 straight into GPU memory, so the model needs no `/tmp` space and no download
step. Without it the image downloads the model to `/tmp` first — size `EphemeralStorage` accordingly. Pointing `MODEL_ID` at a Hugging Face model ID
works for a first test but repeats the download on every new execution environment.

Build and push to your own ECR repository:

```bash
aws ecr create-repository --repository-name my-gpu-function --region us-west-2
aws ecr get-login-password --region us-west-2 \
  | docker login --username AWS --password-stdin 111122223333.dkr.ecr.us-west-2.amazonaws.com

docker build -t my-gpu-function .
docker tag my-gpu-function:latest 111122223333.dkr.ecr.us-west-2.amazonaws.com/my-gpu-function:v1
docker push 111122223333.dkr.ecr.us-west-2.amazonaws.com/my-gpu-function:v1
```

Create the function, publish a version, and invoke it. Publishing is what loads the model: the version reaches `Active` only once the server is ready.

```bash
aws lambda create-function \
  --function-name my-gpu-function \
  --package-type Image \
  --code ImageUri=111122223333.dkr.ecr.us-west-2.amazonaws.com/my-gpu-function:v1 \
  --role arn:aws:iam::111122223333:role/my-gpu-function-role \
  --memory-size 16384 \
  --timeout 900 \
  --accelerator-config '{"AcceleratorMemorySize":12}' \
  --capacity-provider-config '{"LambdaManagedInstancesCapacityProviderConfig":{"CapacityProviderArn":"arn:aws:lambda:us-west-2:111122223333:capacity-provider:my-gpu-capacity-provider","ExecutionEnvironmentMemoryGiBPerVCpu":4}}'

aws lambda publish-version --function-name my-gpu-function
aws lambda get-function-configuration --function-name my-gpu-function:1 --query State
```

The event is the OpenAI request body. A payload with `messages` is routed to `/v1/chat/completions`, one with `prompt` to `/v1/completions`.

```bash
aws lambda invoke \
  --function-name my-gpu-function:1 \
  --payload '{"messages":[{"role":"user","content":"What is a GPU?"}],"max_tokens":64}' \
  response.json
```

!!! note "Size vCPUs, not just memory"
    An execution environment's vCPUs are carved out of the instance at `ExecutionEnvironmentMemoryGiBPerVCpu` GiB each. At the default ratio of 2,
    `--memory-size 16384` asks for 8 vCPUs — every vCPU on a `g6.2xlarge` — and the version fails to activate with
    `IncompatibleAcceleratorInstanceTypes`. Raising the ratio to 4 asks for 4 vCPUs and fits.

For the models validated against each engine, see the [vLLM](../../vllm/models/index.md) and [SGLang](../../sglang/models/index.md) lists. Other
models supported upstream, including embedding and vision-language models, generally work too.

## Write your own handler

Use the Base, CuPy, or PyTorch image, or override the default handler on vLLM and SGLang:

```dockerfile
FROM public.ecr.aws/lambda/deep-learning-containers/pytorch:cuda-v1

COPY handler.py /var/task/
CMD ["handler.handler"]
```

### Concurrency and GPU state

The RIC imports your module once per execution environment, then forks one worker process per concurrent invocation. Workers inherit Python objects
but **not GPU state** — a CUDA context, an EGL context, or a model moved to the GPU before the fork is unusable in a worker. Two patterns work:

**Move to the GPU per worker.** Load weights into CPU memory at import time, then move them on each worker's first invocation.

```python
import os, torch

model = load_model_to_cpu()   # once per execution environment
model.eval()

_gpu_pid = None

def _ensure_on_gpu():
    global _gpu_pid
    if _gpu_pid != os.getpid():
        model.cuda()
        _gpu_pid = os.getpid()

def handler(event, context):
    _ensure_on_gpu()
    with torch.inference_mode():
        return run_inference(model, event)
```

Each worker then holds its own CUDA context and its own copy on the GPU, so GPU memory scales with concurrency. Fine for small models; for a large
one, set `PerExecutionEnvironmentMaxConcurrency` to 1 or use the next pattern.

**One shared server.** Start a server process from the pre-fork hook, which runs once in the parent, and proxy to it from the handler. One copy of the
weights serves all workers. This is what the vLLM and SGLang images do.

```python
from awslambdaric.lambda_concurrency_hooks import register_pre_fork

@register_pre_fork
def start_server():
    ...  # launch the server, then poll its health endpoint until ready

def handler(event, context):
    return requests.post("http://127.0.0.1:8000/predict", json=event).json()
```

A failure inside the hook surfaces as an initialization error (`Runtime.PreForkError`), so no worker starts against a server that never came up.

### Loading weights from S3

The PyTorch, vLLM, and SGLang images include `s3torchconnector`; vLLM and SGLang also include the Run:ai Model Streamer. Streaming avoids writing the
weights to `/tmp` and reading them back, which matters above roughly 1 GB and is the only option when the weights exceed `/tmp`.

```python
from s3torchconnector import S3Checkpoint

checkpoint = S3Checkpoint(region="us-west-2")
with checkpoint.reader("s3://amzn-s3-demo-bucket/models/model.pt") as reader:
    state_dict = torch.load(reader, map_location="cpu", weights_only=True)
```

Keep the bucket in the same Region as the function.

## Media and graphics

The GPUs' NVENC and NVDEC engines are separate from the CUDA cores, so transcoding needs no CUDA code. The PyTorch image's FFmpeg is built with
NVIDIA hardware acceleration:

```bash
ffmpeg -y -hwaccel cuda -hwaccel_output_format cuda -i input.mp4 \
  -c:v h264_nvenc -preset p4 -b:v 5M -c:a copy output.mp4
```

`-hwaccel_output_format cuda` keeps decoded frames in GPU memory so they reach the encoder without a round trip through CPU memory. Request 16 GB or
more of accelerator memory to land on a whole GPU: G6f and Gr6f fractions expose the encode and decode engines but share them with other instances on
the same physical GPU.

Concurrent invocations in one execution environment share `/tmp`, so give each invocation its own working directory:

```python
with tempfile.TemporaryDirectory(dir="/tmp") as workdir:
    ...
```

## Startup time

Publishing a version pulls the image, starts the container, and runs your initialization code. For GPU functions the image pull usually dominates.

- **Keep weights out of the image.** Load them from S3 at initialization instead. Changing models then needs no rebuild.
- **Spread content across layers.** {{ lambda_short }} downloads layers concurrently and unpacks each as it arrives, so one very large layer unpacks
  more slowly than the same content split across several. Give each model shard its own `COPY`. Do not flatten the image.
- **Keep single files under 10 GB.** A larger file fails activation. Images themselves can reach 32 GB, and `/tmp` 32 GB.
- **Consider SOCI.** A SOCI index lets an execution environment start before the whole image has downloaded. See
  [lazy loading](https://docs.aws.amazon.com/lambda/latest/dg/lambda-managed-instances.html) in the {{ lambda_short }} Developer Guide.
- **Set the timeout for initialization.** It is the greater of 130 seconds and the function timeout, capped at 900 seconds.

## Run locally

With no {{ lambda_short }} Runtime API in the environment, the entrypoint starts the Runtime Interface Emulator on port 8080:

```bash
docker run --rm --gpus all -p 9000:8080 \
  -e MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct \
  111122223333.dkr.ecr.us-west-2.amazonaws.com/my-gpu-function:v1

curl -s http://localhost:9000/2015-03-31/functions/function/invocations \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```
