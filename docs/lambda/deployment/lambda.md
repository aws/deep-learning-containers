# Lambda Managed Instances Deployment

GPU functions run on [{{ lambda }} Managed Instances](https://docs.aws.amazon.com/lambda/latest/dg/lambda-managed-instances.html) with a GPU capacity
provider. They are deployed as container images.

This page covers the image. To create the capacity provider and the function, and for scaling, networking, quotas, and GPU metrics, see
[Getting started with {{ lambda_short }} Managed Instances](https://docs.aws.amazon.com/lambda/latest/dg/lambda-managed-instances-getting-started.html).
For the function settings that matter to these images, see [Configuration](../configuration.md).

## Serve an LLM with no handler code

The vLLM and SGLang images include a handler, so you only set `MODEL_ID`. An extension starts the engine and loads the model during initialization,
and the handler forwards each invocation to it.

To get started, point `MODEL_ID` at a Hugging Face model.

```dockerfile
FROM public.ecr.aws/lambda/deep-learning-containers/vllm:cuda-v1

ENV MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
```

The image downloads the model from the Hugging Face Hub to `/tmp` during initialization. This requires internet access from the capacity provider's
subnets, and the download repeats for every new execution environment.

For production, copy the model to an S3 bucket in the same Region as your function and load it from there.

```dockerfile
FROM public.ecr.aws/lambda/deep-learning-containers/vllm:cuda-v1

ENV MODEL_ID=s3://amzn-s3-demo-bucket/models/Qwen2.5-1.5B-Instruct/
ENV VLLM_LOAD_FORMAT=runai_streamer
```

With `runai_streamer`, the image streams the model from S3 directly into GPU memory, so it does not stage the weights in `/tmp`. Grant the function's
execution role `s3:GetObject` and `s3:ListBucket` on the prefix.

Build the image and push it to your own {{ ecr_short }} repository, then create the function on your GPU capacity provider with
[CreateFunction](https://docs.aws.amazon.com/lambda/latest/api/API_CreateFunction.html) and publish a version with
[PublishVersion](https://docs.aws.amazon.com/lambda/latest/api/API_PublishVersion.html). Set `Timeout` to 900 and size `AcceleratorMemorySize` for
your model. For more information, see [Configuration](../configuration.md).

Publishing loads the model, so the version becomes active only after the engine answers its health check. Check the state with
[GetFunctionConfiguration](https://docs.aws.amazon.com/lambda/latest/api/API_GetFunctionConfiguration.html), then invoke the function.

The payload is an OpenAI request body. The handler sends it to the engine's chat completions or completions endpoint, and returns the engine's
response.

```bash
aws lambda invoke \
  --function-name my-gpu-function \
  --payload '{"messages":[{"role":"user","content":"What is a GPU?"}],"max_tokens":64}' \
  response.json
```

For the models tested with each engine, see the [vLLM](../../vllm/models/index.md) and [SGLang](../../sglang/models/index.md) lists. Other models
supported upstream, including embedding and vision-language models, generally work as well.

## The engine extension

On the vLLM and SGLang images, a {{ lambda_short }} extension at `/opt/extensions/<engine>-engine` starts and supervises the engine. The handler does
not launch it.

<div class="wrap-table" markdown>

| Phase | What the extension does |
| --- | --- |
| Initialization | Starts the OpenAI-compatible server on `127.0.0.1` and waits for its health check. Initialization does not complete until the server is ready, so no invocation reaches a server that cannot serve it. If the server fails to start, the extension reports an initialization error and exits, and {{ lambda_short }} discards the environment. |
| Serving | A watchdog probes the health endpoint every `ENGINE_HEALTH_INTERVAL` seconds. After `ENGINE_HEALTH_FAILURES` consecutive failures, or as soon as the server process exits, the extension reports an error and exits so that {{ lambda_short }} replaces the environment. |
| Shutdown | Stops the server and its worker processes so that none of them is left holding GPU memory. |

</div>

An engine that crashes or stops responding therefore costs one execution environment rather than a stream of failed invocations, and your handler does
not have to detect it.

You can tune the server with `MODEL_ID`, the `VLLM_*` and `SGLANG_*` variables, and `VLLM_SERVER_ARGS` or `SGLANG_SERVER_ARGS` for any flag the other
variables do not cover. The extension reads all of them, so none of this requires replacing the handler. For more information, see
[Configuration](../configuration.md).

### Running your own server instead

Set `ENGINE_EXTENSION=off` and the extension registers without starting an engine. The execution environment still waits for the extension during
initialization, but nothing supervises your server, so you own its startup, readiness, and failure handling.

Do this only if you need a server the extension cannot start, such as a server that is not OpenAI-compatible or a custom engine. To change flags or
models, use `*_SERVER_ARGS` and `MODEL_ID` and leave the extension on.

With the extension off, every forked worker imports your handler, so elect one worker to launch the server. The engine images include an
`execute_once` decorator at `/var/task/execute_once.py` for this purpose.

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

The first worker to create `lock_path` runs the function and writes `result_path`, and the others wait for that file until `timeout`. Call `_start()`
at module scope rather than from the handler. The import runs during initialization, so the server is ready before the first invocation arrives, and a
failure is reported as an initialization error. Each worker still waits on the health endpoint itself, because the lock is released when the launch
call returns rather than when the server is ready.

## Write your own handler

A handler is the function code {{ lambda_short }} runs on each invocation.

Among the five images, Base, CuPy, and PyTorch do not include one, so you add your own. vLLM and SGLang include a default handler, which forwards each
request to the engine and returns its response. That is enough to serve a model with no code.

Your own handler can add:

* **Request validation** – Cap `max_tokens`, restrict which models callers may request, or reject malformed input.
* **Prompt templating** – Add a system prompt, or build the prompt from your own request format.
* **Response shaping** – Return the fields your clients expect instead of the engine's output.
* **Pre- and post-processing** – Tokenize, filter, enrich, or log around the inference call.

On vLLM and SGLang, your handler replaces only the request path. The extension still owns the engine, its health checks, and its recovery. To start
the server yourself instead, see [Running your own server instead](#running-your-own-server-instead).

### The contract

All five images set `WORKDIR /var/task` and `CMD ["handler.handler"]`, which the RIC reads as the `handler` function in the `handler` module. A file
at `/var/task/handler.py` that defines `handler(event, context)` is therefore picked up with no further configuration.

```dockerfile
FROM public.ecr.aws/lambda/deep-learning-containers/pytorch:cuda-v1

COPY handler.py /var/task/
```

On vLLM and SGLang, this replaces the handler the image ships. To keep both, use a different module name and point `CMD` at it.

```dockerfile
FROM public.ecr.aws/lambda/deep-learning-containers/vllm:cuda-v1

COPY my_handler.py /var/task/
CMD ["my_handler.handler"]
```

`event` is the invocation payload. The RIC can pass it as a `dict` or as raw `bytes`, so normalize it before use.

```python
import json

if isinstance(event, (bytes, bytearray, str)):
    event = json.loads(event or "{}")
```

### Customize the entrypoint

Each image's `ENTRYPOINT` is a small shim: `/lambda_entrypoint.sh` on Base, CuPy, and PyTorch, and `/vllm_entrypoint.sh` or `/sglang_entrypoint.sh` on
the engine images. It runs the RIC on {{ lambda_short }} and the Runtime Interface Emulator locally.

```sh
if [ -z "${AWS_LAMBDA_RUNTIME_API}" ]; then
  exec /usr/local/bin/aws-lambda-rie "$@"
else
  exec "$@"
fi
```

Replace it when you need work done before the runtime starts, such as staging an asset, setting a variable from instance metadata, or warming a cache.
Keep the branch and both `exec` calls. Without them you lose local `docker run` support and leave an extra process between {{ lambda_short }} and the
RIC.

```dockerfile
FROM public.ecr.aws/lambda/deep-learning-containers/pytorch:cuda-v1

COPY my_entrypoint.sh /my_entrypoint.sh
RUN chmod +x /my_entrypoint.sh
ENTRYPOINT ["/my_entrypoint.sh", "python", "-m", "awslambdaric"]
```

For anything that only needs to run once per execution environment, prefer your handler's module scope. It already runs once per worker, and errors
there are reported as initialization failures rather than as a container that exits.

### Concurrency and GPU state

The RIC forks one worker process per concurrent invocation, and each worker imports your handler itself. Module-level code therefore runs once per
worker rather than once per execution environment, so whatever you load there exists as many times as the concurrency you configured. You can use
either of the following patterns.

**A model per worker.** Load the weights at module scope and move them to the GPU there. Each worker gets its own process, its own CUDA context, and
its own copy of the model.

```python
import torch

model = load_model()          # runs in every worker
model.eval()
model.cuda()

def handler(event, context):
    with torch.inference_mode():
        return run_inference(model, event)
```

GPU memory scales with `PerExecutionEnvironmentMaxConcurrency` because each worker loads the handler itself, so at the default of 4 you need room for
four copies. This works for a small model. For a large one, set the concurrency to 1 or use the next pattern.

**One shared server.** Run the model in one process outside the handler and have each worker send requests to it over loopback, so a single copy of
the weights serves every invocation. The vLLM and SGLang images already do this. For more information, see
[The engine extension](#the-engine-extension). On Base, CuPy, or PyTorch, start the server from your own {{ lambda_short }} extension. Anything
executable in `/opt/extensions` runs before the runtime, and initialization does not finish until the extension asks for its first event, so no
invocation arrives before the model is loaded.

## Loading weights from S3

The PyTorch, vLLM, and SGLang images include `s3torchconnector`, and the vLLM and SGLang images also include the Run:ai Model Streamer. Streaming the
weights avoids writing them to `/tmp` and reading them back, which matters above roughly 1 GB and is the only option when the weights are larger than
`/tmp`.

```python
from s3torchconnector import S3Checkpoint

checkpoint = S3Checkpoint(region="us-west-2")
with checkpoint.reader("s3://amzn-s3-demo-bucket/models/model.pt") as reader:
    state_dict = torch.load(reader, map_location="cpu", weights_only=True)
```

Keep the bucket in the same Region as the function.

## Media and graphics

The NVENC and NVDEC engines on these GPUs are separate from the CUDA cores, so transcoding does not require CUDA code. The PyTorch image's FFmpeg is
built with NVIDIA hardware acceleration.

```bash
ffmpeg -y -hwaccel cuda -hwaccel_output_format cuda -i input.mp4 \
  -c:v h264_nvenc -preset p4 -b:v 5M -c:a copy output.mp4
```

`-hwaccel_output_format cuda` keeps decoded frames in GPU memory, so they reach the encoder without a round trip through CPU memory. Request 16 GB or
more of accelerator memory to run on a whole GPU. G6f and Gr6f instances expose the encode and decode engines but share them with other instances on
the same physical GPU.

Concurrent invocations in one execution environment share `/tmp`, so give each invocation its own working directory.

```python
import tempfile

with tempfile.TemporaryDirectory(dir="/tmp") as workdir:
    ...
```

## Startup time

Publishing a version pulls the image, starts the container, and runs initialization, which on the engine images includes loading the model. The image
pull usually takes the longest. Initialization is limited to 900 seconds, and a version that exceeds it does not become active.

* **Keep weights out of the image** – Load them from S3 during initialization instead. Changing models then requires no rebuild.
* **Spread content across layers** – {{ lambda_short }} downloads layers concurrently and unpacks each one as it arrives, so a single very large layer
  unpacks more slowly than the same content split across several. Give each model shard its own `COPY` instruction, and do not flatten the image.
* **Keep single files under 10 GB** – A larger file fails activation. Images can reach 32 GB, as can `/tmp`.
* **Consider SOCI** – A SOCI index lets an execution environment start before the whole image has downloaded. For more information, see
  [AWS Lambda Seekable OCI (SOCI)](https://aws.amazon.com/about-aws/whats-new/2026/10/aws-lambda-seekable-oci-soci/).
* **Raise both timeouts** – Set `Timeout` to at least 900, and `VLLM_SERVER_TIMEOUT` or `SGLANG_SERVER_TIMEOUT` to just under 900. Their default of
  600 would otherwise stop the model loading first.

## Run locally

When the {{ lambda_short }} Runtime API is not present, the entrypoint starts the Runtime Interface Emulator on port 8080.

```bash
docker run --rm --gpus all -p 9000:8080 \
  -e MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct \
  my-gpu-function:latest

curl -s http://localhost:9000/2015-03-31/functions/function/invocations \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```
