# GPU Functions using Lambda DLC

GPU container base images for [{{ lambda }} Managed Instances](https://docs.aws.amazon.com/lambda/latest/dg/lambda-managed-instances.html). Each image
bundles CUDA, a GPU stack, and the {{ lambda_short }} Runtime Interface Client (RIC), so you add a handler and deploy — no driver or RIC setup. Built
on Amazon Linux 2023 with ongoing security patching by {{ aws }}.

These images target {{ lambda_short }} only. They ship a {{ lambda_short }} entrypoint and expect the {{ lambda_short }} Runtime API, so they do not
serve over a port like the {{ ec2_short }} and {{ sagemaker }} DLCs. For those platforms, use the [vLLM](../vllm/index.md),
[SGLang](../sglang/index.md), [PyTorch](../pytorch/index.md), or [Base](../base/index.md) images instead.

## Images

| Image | Tag | What's added on top of Base |
| --- | --- | --- |
| Base | `base:cuda` | CUDA runtime, Python, RIC — nothing else |
| CuPy | `cupy:cuda` | CuPy (NumPy-compatible GPU arrays), NumPy |
| PyTorch | `pytorch:cuda` | PyTorch, torchvision, torchaudio, Transformers, Diffusers, s3torchconnector, NVENC-enabled FFmpeg |
| vLLM | `vllm:cuda` | vLLM server + engine extension + default handler, FlashInfer with prebuilt kernels, Run:ai Model Streamer, on the PyTorch stack |
| SGLang | `sglang:cuda` | SGLang server + engine extension + default handler, sgl-kernel, FlashInfer, Run:ai Model Streamer, on the PyTorch stack |

All five are x86-64, GPU-only, CUDA 13.0.3, Python 3.13, `awslambdaric` 4.2.2.

**Public:** `public.ecr.aws/lambda/deep-learning-containers/<image>:<tag>` — one repository per image, published in the
[{{ lambda_short }} ECR Public Gallery](https://gallery.ecr.aws/lambda). Pin a major version with `cuda-v1`, or a patch with `cuda-v1.0.0`.

**Private:** a single `lambda` repository where the image name moves into the tag, for example
`763104351884.dkr.ecr.us-west-2.amazonaws.com/lambda:vllm-cuda`. See [Image Access](../get_started/index.md) for authentication and
[Region Availability](../reference/region_availability.md) for per-region account IDs.

## Choosing an image

- **Base** — custom CUDA kernels, or a framework none of the other images carry (for example Blender or GROMACS). Smallest image.
- **CuPy** — GPU array math without an ML framework: signal processing, Monte Carlo simulation, numerical workloads.
- **PyTorch** — your own model code: classification, detection, diffusion, speech. Also the image for media work, since its FFmpeg is built with NVENC
  and NVDEC.
- **vLLM** or **SGLang** — LLM serving with no handler code. Set `MODEL_ID` and deploy; the image runs an OpenAI-compatible server. Pick SGLang for
  structured output and prefix-cache-heavy traffic, vLLM otherwise.

## How the images run on {{ lambda_short }}

Each image declares `ENTRYPOINT ["<image>_entrypoint.sh", "python", "-m", "awslambdaric"]` and `CMD ["handler.handler"]`. The entrypoint execs the RIC
when the {{ lambda_short }} Runtime API is present, and the Runtime Interface Emulator when it is not, so the same image runs locally under
`docker run`. Both are replaceable — see [Write your own handler](deployment/lambda.md#write-your-own-handler) and
[Customize the entrypoint](deployment/lambda.md#customize-the-entrypoint).

**Most functions ship their own handler.** Base, CuPy, and PyTorch contain none: drop a `handler.py` defining `handler(event, context)` into
`/var/task` and it is picked up. vLLM and SGLang do ship one, and it gets the shared-engine structure right, but it forwards requests to the engine
unvalidated and cannot stream — a reference to build on rather than the end state. See
[Write your own handler](deployment/lambda.md#write-your-own-handler) for what it does and does not do.

{{ lambda_short }} gives each execution environment exactly one GPU and runs several concurrent invocations in it, governed by
`PerExecutionEnvironmentMaxConcurrency` (default 4, maximum 48). The RIC runs your initialization code once in a parent process and then forks one
worker per concurrent invocation. **A CUDA context does not survive the fork**, so in your own handler, move the model to the GPU inside the handler
on each worker's first invocation. See [Concurrency and GPU state](deployment/lambda.md#concurrency-and-gpu-state).

On the vLLM and SGLang images a **{{ lambda_short }} extension owns the engine**, not the handler. It starts the server during Init and holds the
execution environment out of service until `/health` answers, then keeps probing: if the server dies or stops answering, the extension exits and
{{ lambda_short }} replaces the environment. One copy of the weights serves every concurrent invocation and the handler is only an HTTP proxy. See
[The engine extension](deployment/lambda.md#the-engine-extension).

`/tmp` is the only writable path, so the images point their caches there: `HOME=/tmp`, `TORCH_HOME=/tmp/torch`, `HF_HOME=/tmp/huggingface`,
`CUPY_CACHE_DIR=/tmp/cupy_cache`.

## How We Build

- **One multi-stage Dockerfile** — all five images build from `docker/lambda/Dockerfile`, sharing a CUDA base, a Python build, and a PyTorch builder
  stage, so the stacks cannot drift apart.
- **Pinned and locked** — Python dependencies are resolved with `uv` and committed as lock files per image. vLLM and SGLang are built from a pinned
  upstream commit, with the torch trio constrained to the PyTorch image's exact cu130 versions.
- **Tested before release** — in-container unit tests, single-GPU tests, and an invoke test on real {{ lambda_short }} GPU functions that asserts
  worker topology and that the model occupies the GPU exactly once.
- **Security-patched** — continuous {{ aws }} security patching on Amazon Linux 2023.
