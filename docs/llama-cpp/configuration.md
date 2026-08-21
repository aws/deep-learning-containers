# Configuration

llama.cpp is configured differently on the two platforms:

- **{{ ec2_short }}** — pass native `llama-server` flags as container arguments; a small set of DLC environment variables control auth and the
  upstream port.
- **{{ sagemaker }}** — set `SM_LLAMA_CPP_*` environment variables on the container; the entrypoint translates them into `llama-server` flags.

The complete, authoritative flag list is the upstream [`llama-server` documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) —
this page covers the DLC-specific surface and the flags most users need.

## EC2 Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `LLAMA_API_KEY` | *(unset)* | If set, passed to `llama-server --api-key`; every request must then carry `Authorization: Bearer <key>`. Unset means the endpoint is **unauthenticated** |

On {{ ec2_short }}, everything else is a `llama-server` argument appended to `docker run` (the host/port are fixed at `0.0.0.0:8080` by the
entrypoint). See [Common llama-server Flags](#common-llama-server-flags).

## SageMaker Environment Variables

The {{ sagemaker }} entrypoint reads a few control variables and then translates **every other** `SM_LLAMA_CPP_*` variable into a `llama-server` flag:
`SM_LLAMA_CPP_FOO_BAR=value` → `--foo-bar value`. A value of `true` becomes a bare flag (`--foo-bar`); a value of `false` is omitted.

| Variable | Default | Description |
| --- | --- | --- |
| `SM_LLAMA_CPP_MODEL` | *(unset)* | Explicit GGUF path → `--model`; skips `/opt/ml/model` auto-detection |
| `SM_LLAMA_CPP_MODEL_DIR` | `/opt/ml/model` | Directory scanned for a `*.gguf` when `SM_LLAMA_CPP_MODEL` is unset |
| `SM_LLAMA_CPP_HF_REPO` | *(unset)* | → `--hf-repo`; HuggingFace repo to download the GGUF from at startup |
| `SM_LLAMA_CPP_HF_FILE` | *(unset)* | → `--hf-file`; GGUF filename within the HuggingFace repo |
| `SM_LLAMA_CPP_CTX_SIZE` | *(llama-server default)* | → `--ctx-size`; context window (tokens) |
| `SM_LLAMA_CPP_PORT` | `8080` | Public port nginx serves on |
| `LLAMA_CPP_UPSTREAM_PORT` | `8081` | Loopback port `llama-server` binds behind nginx |
| `SM_LLAMA_CPP_<ANY>` | *(unset)* | Any other suffix maps to `--<any>` on `llama-server` |

`SM_LLAMA_CPP_MODEL`, `SM_LLAMA_CPP_MODEL_DIR`, and `SM_LLAMA_CPP_PORT` are consumed by the entrypoint itself and are **not** forwarded as flags.

## Common llama-server Flags

These are passed as container arguments on {{ ec2_short }}, or as `SM_LLAMA_CPP_*` variables on {{ sagemaker }} (e.g. `--n-gpu-layers` ↔
`SM_LLAMA_CPP_N_GPU_LAYERS`).

| Flag | Applies to | Description |
| --- | --- | --- |
| `--model <path>` | all | Path to a local GGUF file |
| `--hf-repo <repo>` / `--hf-file <file>` | all | Download a GGUF from HuggingFace at startup |
| `--ctx-size <n>` | all | Context window size in tokens |
| `--n-gpu-layers <n>` | GPU only | Layers to offload to the GPU (`999` = all); no effect on CPU / Graviton images |
| `--parallel <n>` | all | Number of parallel request slots |
| `--threads <n>` | all | CPU threads for generation |
| `--batch-size <n>` | all | Logical batch size |
| `--api-key <key>` | all | Require a bearer token (on {{ ec2_short }}, prefer the `LLAMA_API_KEY` env var) |

The DLC sets **no defaults** for concurrency, threads, batch size, or context size beyond llama-server's own — tune them for your model and instance.

## Known Limitations

- **GGUF only.** The server loads GGUF-format models; other formats must be converted first (see [Supported Models](models/index.md)).
- **No baked-in model.** You must supply a GGUF at launch via a local file, `/opt/ml/model`, or a HuggingFace download.
- **Unauthenticated by default on {{ ec2_short }}.** The endpoint binds `0.0.0.0:8080` with no auth unless `LLAMA_API_KEY` is set — run it inside a
  private network. The {{ sagemaker }} path is gated by SageMaker's own request authentication.
- **HuggingFace download needs network egress.** The `--hf-repo` / `SM_LLAMA_CPP_HF_REPO` path is not air-gapped; mount the GGUF locally to run
  offline.
- **GPU on {{ sagemaker }} needs a recent driver AMI.** The GPU image ships CUDA 13.0.2; pin a recent `InferenceAmiVersion` — see
  [SageMaker Deployment](deployment/sagemaker.md#notes).

## Full Reference

- [`llama-server` documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
