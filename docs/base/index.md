# Build Custom Images using Base DLC

Lightweight Base Docker images with NVIDIA CUDA and Python pre-installed on Amazon Linux 2023. Use them as the `FROM` for your own AI/ML
images, or as a quick interactive environment for prototyping. Built and patched continuously by {{ aws }}.

## Images

| Variant | Image | CUDA | Use Case |
| --- | --- | --- | --- |
| Runtime, CUDA 12.9 | `public.ecr.aws/deep-learning-containers/base:runtime-cuda-v1` | 12.9.1 | Run a CUDA application — minimal size, no compilers |
| Devel, CUDA 12.9 | `public.ecr.aws/deep-learning-containers/base:devel-cuda-v1` | 12.9.1 | Compile CUDA code — adds `nvcc`, headers, gcc, cmake |
| Runtime, CUDA 13.0 | `public.ecr.aws/deep-learning-containers/base:runtime-cuda-v2` | 13.0.2 | Same as v1 but on CUDA 13 |
| Devel, CUDA 13.0 | `public.ecr.aws/deep-learning-containers/base:devel-cuda-v2` | 13.0.2 | Same as v1 but on CUDA 13 |

All images are also available on the [ECR Public Gallery](https://gallery.ecr.aws/deep-learning-containers/base). For private ECR URIs, see
[Image Access](../get_started/index.md).

## What's Included

All four variants share the same core stack:

- **Amazon Linux 2023** with continuous security patching
- **Python 3.13.12** built from source with hardening flags, available as `python` / `python3`
- **uv** package manager, pre-installed at `/usr/local/bin/uv`
- **NVIDIA CUDA Toolkit** (12.9.1 for v1, 13.0.2 for v2), based on the upstream `nvidia/cuda:*-amzn2023` images

The **devel** variants additionally include the full CUDA Toolkit (`nvcc`, headers, libraries) plus `gcc`, `gcc-c++`, `cmake`, `automake`, `autoconf`,
`git`, `make`, and `tar` — sufficient for compiling CUDA C/C++ code or wheels with native extensions.

The **runtime** variants ship only the CUDA runtime libraries needed to *run* CUDA applications, keeping image size minimal for production
deployments.

## Use as a Base for Your Own Image

The most common pattern is to use these images as the `FROM` line in your own Dockerfile. Build wheels and compile native code in `devel`, then copy
the result into `runtime` for a small final image:

```dockerfile
# Stage 1: build wheels in the devel image
FROM public.ecr.aws/deep-learning-containers/base:devel-cuda-v2 AS build
WORKDIR /app
COPY requirements.txt .
RUN pip install --target=/app/deps -r requirements.txt

# Stage 2: minimal runtime
FROM public.ecr.aws/deep-learning-containers/base:runtime-cuda-v2
COPY --from=build /app/deps /app/deps
ENV PYTHONPATH=/app/deps
COPY app.py /app/
CMD ["python", "/app/app.py"]
```

If your project doesn't need to compile anything, just use the runtime image directly:

```dockerfile
FROM public.ecr.aws/deep-learning-containers/base:runtime-cuda-v2
RUN pip install transformers torch
COPY . /app
WORKDIR /app
CMD ["python", "main.py"]
```

## Run Interactively

For quick prototyping or one-off CUDA work, you can run the image directly:

```bash
docker run --rm -it --gpus all \
  public.ecr.aws/deep-learning-containers/base:devel-cuda-v2 \
  bash
```

Inside the container, `python`, `pip`, `uv`, and (in the devel variant) `nvcc` are on the `PATH`. CUDA libraries are installed at `/usr/local/cuda`.

## Choosing v1 vs v2

- **v1 (CUDA 12.9):** use when your stack is pinned to CUDA 12.x — most current PyTorch wheels (cu128/cu129) are compatible.
- **v2 (CUDA 13.0):** use when you want CUDA 13.x compatibility (e.g., to match the vLLM-Omni DLC, which is built on CUDA 13). Some older NVIDIA
  drivers may need the bundled `cuda-compat-13-0` forward-compat layer to run CUDA 13 binaries.

CUDA major versions are not interchangeable at runtime — pick the variant that matches the GPU drivers on your target hosts.

## How We Build

These images are curated builds:

- **Built from upstream `nvidia/cuda:*-amzn2023` images** — we add Python compiled from source, OSS license metadata, and security patches.
- **Continuously patched** — security updates from {{ aws }} and NVIDIA are applied on every build.
- **Versioned** — major version (`v1`, `v2`) tracks the CUDA major. New CUDA majors get a new tag suffix; existing tags stay on their pinned CUDA
  release across minor patches.
