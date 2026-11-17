# GPU Functions using Lambda DLC

GPU container base images for [{{ lambda }} Managed Instances](https://docs.aws.amazon.com/lambda/latest/dg/lambda-managed-instances.html). Each image
includes CUDA, a GPU stack, and the {{ lambda_short }} Runtime Interface Client (RIC), so you add your handler and deploy without installing a driver
or configuring the runtime. The images are built on Amazon Linux 2023 and patched continuously by {{ aws }}.

These images are for {{ lambda_short }} only. They ship a {{ lambda_short }} entrypoint and expect the {{ lambda_short }} Runtime API, so they do not
serve over a port the way the {{ ec2_short }} and {{ sagemaker }} DLCs do. For those platforms, use the [vLLM](../vllm/index.md),
[SGLang](../sglang/index.md), [PyTorch](../pytorch/index.md), or [Base](../base/index.md) images.

## Images

<div class="wrap-table" markdown>

| Image | Tag | What's added on top of Base |
| --- | --- | --- |
| Base | `base:cuda` | CUDA runtime, Python, and the RIC |
| CuPy | `cupy:cuda` | CuPy (NumPy-compatible GPU arrays), NumPy |
| PyTorch | `pytorch:cuda` | PyTorch, torchvision, torchaudio, Transformers, Diffusers, s3torchconnector, NVENC-enabled FFmpeg |
| vLLM | `vllm:cuda` | vLLM server, engine extension, default handler, FlashInfer with prebuilt kernels, Run:ai Model Streamer, on the PyTorch stack |
| SGLang | `sglang:cuda` | SGLang server, engine extension, default handler, sgl-kernel, FlashInfer, Run:ai Model Streamer, on the PyTorch stack |

</div>

All five images are x86-64 and GPU-only, with CUDA 13.0.3, Python 3.13, and `awslambdaric` 4.2.2.

Pull from `public.ecr.aws/lambda/deep-learning-containers/<image>:<tag>`. Each image has its own repository in the
[{{ lambda_short }} ECR Public Gallery](https://gallery.ecr.aws/lambda). Pin a major version with `cuda-v1`, or a patch with `cuda-v1.0.0`. For
private
{{ ecr_short }} URIs, see [Image Access](../get_started/index.md).

## Choosing an image

* **Base** – Use for custom CUDA kernels, or for a framework the other images do not carry, such as Blender or GROMACS. This is the smallest image.
* **CuPy** – Use for GPU array math without an ML framework, such as signal processing, Monte Carlo simulation, and other numerical workloads.
* **PyTorch** – Use for your own model code, such as classification, detection, diffusion, and speech. This is also the image for media work, because
  its FFmpeg is built with NVENC and NVDEC.
* **vLLM** or **SGLang** – Use for LLM serving. Set `MODEL_ID` and deploy, and the image runs an OpenAI-compatible server. Both engines support
  structured output and prefix caching. SGLang's RadixAttention is designed for workloads that reuse long prefixes, such as multi-turn chat and agent
  loops. Benchmark both for your workload.

## How the images run on {{ lambda_short }}

Each image declares `ENTRYPOINT ["<image>_entrypoint.sh", "python", "-m", "awslambdaric"]` and `CMD ["handler.handler"]`. The entrypoint runs the RIC
when the {{ lambda_short }} Runtime API is present, and the Runtime Interface Emulator when it is not, so the same image runs locally under
`docker run`. You can replace either one. For more information, see [Write your own handler](deployment/lambda.md#write-your-own-handler) and
[Customize the entrypoint](deployment/lambda.md#customize-the-entrypoint).

Most functions ship their own handler. Base, CuPy, and PyTorch do not include one, so add a `handler.py` that defines `handler(event, context)` to
`/var/task` and {{ lambda_short }} picks it up. The vLLM and SGLang images do include a handler, but it forwards requests to the engine without
validating them and it cannot stream, so treat it as a starting point. For more information, see
[Write your own handler](deployment/lambda.md#write-your-own-handler).

{{ lambda_short }} assigns one GPU to each execution environment and runs several concurrent invocations in it, controlled by
`PerExecutionEnvironmentMaxConcurrency` (default 4, maximum 48). The RIC forks one worker per concurrent invocation, and each worker imports your
handler itself, so module-level code runs once per worker. A model loaded there exists once per worker, so at the default concurrency four copies must
fit on the GPU. For more information, see [Concurrency and GPU state](deployment/lambda.md#concurrency-and-gpu-state).

On the vLLM and SGLang images, a {{ lambda_short }} extension owns the engine rather than the handler. The extension starts the server during
initialization and keeps the execution environment out of service until the server answers its health check, then probes it for as long as the
environment lives. If the server stops answering, the extension exits and {{ lambda_short }} replaces the environment. One copy of the weights serves
every concurrent invocation, and the handler only proxies requests to it. For more information, see
[The engine extension](deployment/lambda.md#the-engine-extension).

`/tmp` is the only writable path, so the images point their caches there with `HOME=/tmp`, `TORCH_HOME=/tmp/torch`, `HF_HOME=/tmp/huggingface`, and
`CUPY_CACHE_DIR=/tmp/cupy_cache`.

## How We Build

* **One multi-stage Dockerfile** – All five images build from `docker/lambda/Dockerfile` and share a CUDA base, a Python build, and a PyTorch builder
  stage, so the stacks cannot drift apart.
* **Pinned and locked** – Python dependencies are resolved with `uv` and committed as lock files for each image. The vLLM and SGLang images are built
  from a pinned upstream commit, with torch, torchvision, and torchaudio constrained to the PyTorch image's cu130 versions.
* **Tested before release** – Each release runs in-container unit tests, single-GPU tests, and an invoke test on real {{ lambda_short }} GPU functions
  that checks worker topology, that the model occupies the GPU exactly once, and that {{ lambda_short }} replaces the execution environment when the
  engine is killed.
* **Security-patched** – {{ aws }} patches the Amazon Linux 2023 base continuously.
