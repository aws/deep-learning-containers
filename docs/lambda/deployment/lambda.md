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

The vLLM and SGLang images need no handler code. Set `MODEL_ID`; an extension starts the engine and loads the model during Init, and the shipped
handler proxies invocations to it.

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

**Note:** Size vCPUs, not just memory. An execution environment's vCPUs are carved out of the instance at `ExecutionEnvironmentMemoryGiBPerVCpu` GiB
each. At the default ratio of 2, `--memory-size 16384` asks for 8 vCPUs — every vCPU on a `g6.2xlarge` — and the version fails to activate with
`IncompatibleAcceleratorInstanceTypes`. Raising the ratio to 4 asks for 4 vCPUs and fits.

For the models validated against each engine, see the [vLLM](../../vllm/models/index.md) and [SGLang](../../sglang/models/index.md) lists. Other
models supported upstream, including embedding and vision-language models, generally work too.

## The engine extension

On the vLLM and SGLang images the engine is started and supervised by a {{ lambda_short }} extension at `/opt/extensions/<engine>-engine`, with its
source at `/opt/engine-extension/engine_extension.py`. The handler never launches it.

| Phase | What the extension does |
| --- | --- |
| Init | Starts the OpenAI-compatible server on `127.0.0.1` and waits for `/health`. Init does not complete until it finishes, so no invocation reaches a server that is not ready. On failure it reports an init error and exits, and {{ lambda_short }} discards the environment rather than routing to it. |
| Serving | A watchdog probes `/health` every `ENGINE_HEALTH_INTERVAL` seconds. After `ENGINE_HEALTH_FAILURES` consecutive failures, or as soon as the server process exits, it reports an exit error and exits so {{ lambda_short }} replaces the environment. |
| Shutdown | Stops the server's whole process group — `SIGTERM`, then `SIGKILL` inside {{ lambda_short }}'s 2-second budget — so no worker is left holding GPU memory. |

A crashed or wedged engine therefore costs one execution environment, not a stream of failed invocations. Keep this behaviour: it is the reason you do
not have to detect a dead engine in your handler.

Tune the server through `MODEL_ID`, the `VLLM_*` / `SGLANG_*` variables, and `VLLM_SERVER_ARGS` / `SGLANG_SERVER_ARGS` for any flag those do not cover
— the extension reads all of them, so none of it requires replacing the handler. See [Configuration](../configuration.md).

### Running your own server instead

Set `ENGINE_EXTENSION=off` to make the extension register and idle without starting anything. The execution environment still waits for it during
Init, and nothing supervises your server, so you own startup, readiness, and failure handling.

Only do this if you need a process the extension cannot produce — a non-OpenAI server, or a custom engine. For different flags or a different model,
use `*_SERVER_ARGS` and `MODEL_ID` instead and leave the extension on.

With the extension off, every forked worker imports your handler, so elect one to launch the server. The engine images ship an `execute_once`
decorator at `/var/task/execute_once.py` for exactly this — unused on the default path, still importable:

```python
import subprocess, time
import requests
from execute_once import execute_once

@execute_once(
    lock_path="/tmp/server.lock",
    result_path="/tmp/server.result",
    timeout=600,
    serializer=lambda obj, f: None,
    deserializer=lambda data: None,
)
def _launch():
    return subprocess.Popen(["my-server", "--port", "8000"])

def _start():
    _launch()                 # one worker launches; the rest return immediately
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        try:
            if requests.get("http://127.0.0.1:8000/health", timeout=5).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError("server did not become ready")

_start()                      # at import, so the server is up before the first invocation

def handler(event, context):
    return requests.post("http://127.0.0.1:8000/predict", json=event).json()
```

The first worker to create `lock_path` with `O_EXCL` runs the function and writes `result_path`; the others poll for it until `timeout`. Call this at
module scope, not from the handler: the import happens during Init, so the server is ready before any invocation arrives, and a failure is reported as
an initialization error. Every worker must still wait on the health endpoint itself — the lock is released when the launch call returns, not when the
server is ready.

## Write your own handler

Most functions do. Base, CuPy, and PyTorch ship no handler at all. On vLLM and SGLang the extension owns the engine, so the default handler is only a
proxy: it forwards the event to the engine as an OpenAI request and returns the body. Replacing it leaves the engine, its health checks, and the
recycling behaviour untouched. Take it as the starting point rather than the finished article:

- **No streaming.** The handler does one blocking request and returns the parsed body, so `"stream": true` fails. Lambda invocations are
  request/response; token-by-token streaming is not available through this path.
- **No validation or caps.** The event is forwarded as-is, so a caller sets `max_tokens`, `model`, and every other engine parameter.
- **Engine errors become function errors.** A rejected request raises, so it counts against the function's `Errors` metric and is retried on
  asynchronous invokes instead of being returned to the caller as a 4xx.
- **The response is the raw engine body.** Reshape it here if your clients do not speak OpenAI.
- **`MODEL_ID` defaults to a 0.5B smoke-test model.** Always set it.

### The contract

All five images set `WORKDIR /var/task` and `CMD ["handler.handler"]`, which the RIC reads as *module* `handler`, *function* `handler`. So a file at
`/var/task/handler.py` defining `handler(event, context)` is picked up with no further configuration:

```dockerfile
FROM public.ecr.aws/lambda/deep-learning-containers/pytorch:cuda-v1

COPY handler.py /var/task/
```

On vLLM and SGLang this overwrites the shipped `/var/task/handler.py`. To keep the default alongside yours, use a different module name and point
`CMD` at it:

```dockerfile
FROM public.ecr.aws/lambda/deep-learning-containers/vllm:cuda-v1

COPY my_handler.py /var/task/
CMD ["my_handler.handler"]
```

`event` is the invocation payload. The RIC may hand it to you as a `dict` or as raw `bytes`, so normalize before use:

```python
import json

if isinstance(event, (bytes, bytearray, str)):
    event = json.loads(event or "{}")
```

### Customize the entrypoint

Each image's `ENTRYPOINT` is a small shim — `/lambda_entrypoint.sh` on Base, CuPy, and PyTorch, `/vllm_entrypoint.sh` and `/sglang_entrypoint.sh` on
the engines. It execs the RIC under {{ lambda_short }} and the Runtime Interface Emulator locally:

```sh
if [ -z "${AWS_LAMBDA_RUNTIME_API}" ]; then
  exec /usr/local/bin/aws-lambda-rie "$@"
else
  exec "$@"
fi
```

Replace it when you need work done before the runtime starts — staging an asset, setting a variable from instance metadata, warming a cache. Keep the
branch and both `exec` calls, or you lose local `docker run` support and leave an extra process between {{ lambda_short }} and the RIC:

```dockerfile
FROM public.ecr.aws/lambda/deep-learning-containers/pytorch:cuda-v1

COPY my_entrypoint.sh /my_entrypoint.sh
RUN chmod +x /my_entrypoint.sh
ENTRYPOINT ["/my_entrypoint.sh", "python", "-m", "awslambdaric"]
```

Prefer the handler's module scope for anything that only needs to run once per execution environment — it already does, and errors there are reported
as initialization failures rather than a container that exits.

### Concurrency and GPU state

The RIC forks one worker process per concurrent invocation, and **each worker imports your handler itself**. Module-level code therefore runs once per
worker, not once per execution environment: whatever you load there exists as many times as the concurrency you configured. Nothing is shared between
workers, so there is no inherited CUDA context to work around — but there is also no single copy of your model. Two patterns work:

**A model per worker.** Load the weights at module scope and move them to the GPU there. Each worker gets its own process, its own CUDA context, and
its own copy.

```python
import torch

model = load_model()          # runs in every worker
model.eval()
model.cuda()

def handler(event, context):
    with torch.inference_mode():
        return run_inference(model, event)
```

GPU memory scales with `PerExecutionEnvironmentMaxConcurrency` because each worker loads the handler itself: at the default of 4, four copies have to
fit. Fine for a small model. For a large one, set the concurrency to 1, or use the next pattern.

**One shared server.** Run the model in one process outside the handler and proxy to it over loopback, so a single copy of the weights serves every
worker. On vLLM and SGLang this is already done for you — see [The engine extension](#the-engine-extension). On Base, CuPy, or PyTorch, start the
server from a {{ lambda_short }} extension of your own: anything executable in `/opt/extensions` runs before the runtime, and Init does not finish
until it asks for its first event, so no invocation arrives before the model is loaded.

## Loading weights from S3

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

The GPUs' NVENC and NVDEC engines are separate from the CUDA cores, so transcoding needs no CUDA code. The PyTorch image's FFmpeg is built with NVIDIA
hardware acceleration:

```bash
ffmpeg -y -hwaccel cuda -hwaccel_output_format cuda -i input.mp4 \
  -c:v h264_nvenc -preset p4 -b:v 5M -c:a copy output.mp4
```

`-hwaccel_output_format cuda` keeps decoded frames in GPU memory so they reach the encoder without a round trip through CPU memory. Request 16 GB or
more of accelerator memory to land on a whole GPU: G6f and Gr6f fractions expose the encode and decode engines but share them with other instances on
the same physical GPU.

Concurrent invocations in one execution environment share `/tmp`, so give each invocation its own working directory:

```python
import tempfile

with tempfile.TemporaryDirectory(dir="/tmp") as workdir:
    ...
```

## Startup time

Publishing a version pulls the image, starts the container, and runs Init — on the engine images that includes loading the model. The image pull
usually dominates, and the whole of Init shares one budget: the greater of 130 seconds and the function `Timeout`, capped at 900 seconds. A version
that exceeds it never goes `Active`.

- **Keep weights out of the image.** Load them from S3 at initialization instead. Changing models then needs no rebuild.
- **Spread content across layers.** {{ lambda_short }} downloads layers concurrently and unpacks each as it arrives, so one very large layer unpacks
  more slowly than the same content split across several. Give each model shard its own `COPY`. Do not flatten the image.
- **Keep single files under 10 GB.** A larger file fails activation. Images themselves can reach 32 GB, and `/tmp` 32 GB.
- **Consider SOCI.** A SOCI index lets an execution environment start before the whole image has downloaded. See
  [lazy loading with SOCI](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html#images-create-soci) in the {{ lambda_short }} Developer
  Guide.
- **Give engine functions the whole budget, in two places.** Set `Timeout` to 900 **and** raise `VLLM_SERVER_TIMEOUT` / `SGLANG_SERVER_TIMEOUT` to
  match. `Timeout` sets the Init budget, but the extension gives up on the model after `*_SERVER_TIMEOUT` seconds — 600 by default — whatever
  `Timeout` allows.

## Run locally

With no {{ lambda_short }} Runtime API in the environment, the entrypoint starts the Runtime Interface Emulator on port 8080:

```bash
docker run --rm --gpus all -p 9000:8080 \
  -e MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct \
  111122223333.dkr.ecr.us-west-2.amazonaws.com/my-gpu-function:v1

curl -s http://localhost:9000/2015-03-31/functions/function/invocations \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```
