# Supported Models

llama.cpp serves models in the **GGUF** format — the quantized single-file format used by the llama.cpp ecosystem. Any model that upstream
`llama-server` can load is supported: Llama, Qwen, Mistral, Gemma, Phi, DeepSeek, and the many other architectures llama.cpp implements, in any
quantization (`Q4_K_M`, `Q5_K_M`, `Q8_0`, `F16`, …).

The container ships **no baked-in model** — you supply a GGUF at launch. There are three ways to provide one:

1. **Mount a local GGUF** — a file on {{ ec2_short }}, or a `model.tar.gz` staged via `ModelDataUrl` on {{ sagemaker }}.
2. **Download from HuggingFace at startup** — point the server at a HuggingFace repo and file, and it downloads the GGUF the first time the container
   boots. This path needs network access.
3. **Pass a model URL** — any URL `llama-server` accepts (the build enables libcurl).

## Getting GGUF Models

Thousands of ready-to-run GGUF models are already published on the [HuggingFace Hub](https://huggingface.co/models?library=gguf). The
[ggml-org](https://huggingface.co/ggml-org) and [bartowski](https://huggingface.co/bartowski) collections are good places to start. If you would
rather convert your own weights, the llama.cpp repository ships the tooling to do it:
[`convert_hf_to_gguf.py`](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py) turns a HuggingFace model into a GGUF file, and
[`llama-quantize`](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize) shrinks it to the quantization you want.

Choose a quantization that fits your hardware. A smaller quant such as `Q4_K_M` uses less memory and runs faster, at a small cost to quality. A larger
one such as `Q8_0` or `F16` keeps more of the original quality but needs more RAM or VRAM.

## Choosing a Hardware Target

| Target | Image repository / tag | Notes |
| --- | --- | --- |
| x86 CPU | `llama-cpp:server-cpu-v1` | Runs on any modern x86 instance and picks the best microarchitecture backend at runtime |
| x86 GPU (CUDA) | `llama-cpp:server-cuda-v1` | Needs an NVIDIA GPU — offload layers with `--n-gpu-layers` (see [Configuration](../configuration.md)) |
| Graviton (ARM64) CPU | `llama-cpp-arm64:server-cpu-v1` | Runs on AWS Graviton3/4, tuned for Neoverse-V1. `--n-gpu-layers` does not apply here |

The `-sagemaker-` tags are the {{ sagemaker }} counterparts of the same three targets.

## Specifying the Model

- **{{ ec2_short }}** — pass the model as a `llama-server` argument. Mount a local GGUF and point `--model` at it, or fetch from HuggingFace with
  `--hf-repo` / `--hf-file`. See [EC2 Deployment](../deployment/ec2.md).
- **{{ sagemaker }}** — the entrypoint resolves the model in this order (see [SageMaker Deployment](../deployment/sagemaker.md#specifying-the-model)):
  1. **`SM_LLAMA_CPP_MODEL`** — an explicit GGUF path inside the container.
  2. **`/opt/ml/model`** — the first `*.gguf` staged via `ModelDataUrl` is auto-detected (searched up to two directory levels deep).
  3. **`SM_LLAMA_CPP_HF_REPO` / `SM_LLAMA_CPP_HF_FILE`** — download the GGUF from HuggingFace at startup.

For a **multi-part (sharded) GGUF**, provide every shard and point the server at the first one (`…-00001-of-0000N.gguf`). llama.cpp finds and loads
the rest automatically.

## Offline / Air-Gapped

The container runs with no network access **only when the GGUF is provided locally** — a mounted file on {{ ec2_short }} or a `model.tar.gz` on
{{ sagemaker }}. The HuggingFace-download path (`--hf-repo` / `SM_LLAMA_CPP_HF_REPO`) requires runtime network egress and is not air-gapped.

## Full Reference

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [`llama-server` documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
- [GGUF format](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
