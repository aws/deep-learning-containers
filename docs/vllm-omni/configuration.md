# Configuration

## EC2 / EKS (`omni-cuda`)

Pass vLLM server arguments directly to `docker run`:

```bash
docker run --gpus all -p 8080:8080 \
  public.ecr.aws/deep-learning-containers/vllm:omni-cuda \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

| Argument | Description | Default |
| --- | --- | --- |
| `--model` | Model ID or path (required) | — |
| `--host` | Bind address | `0.0.0.0` |
| `--port` | Server port | `8080` |
| `--tensor-parallel-size` | Number of GPUs | `1` |
| `--max-model-len` | Maximum sequence length | Model default |
| `--gpu-memory-utilization` | Fraction of GPU memory to use | `0.9` |
| `--enforce-eager` | Disable CUDA graphs | `false` |
| `--trust-remote-code` | Allow custom model code (required for some models) | `false` |

For gated models, pass `-e HF_TOKEN=<your_hf_token>`. On hosts with NVIDIA drivers older than the CUDA 13.0 baseline, also pass
`-e VLLM_ENABLE_CUDA_COMPATIBILITY=1`.

## Amazon SageMaker AI (`omni-sagemaker-cuda`)

The SageMaker image serves on **port 8080** and accepts vLLM flags via `SM_VLLM_*` environment variables. Each variable is converted to the
corresponding vLLM flag (e.g., `SM_VLLM_MAX_MODEL_LEN=4096` → `--max-model-len 4096`). Boolean values follow shell convention: `true` becomes a bare
flag (`SM_VLLM_ENFORCE_EAGER=true` → `--enforce-eager`), and `false` omits the flag entirely.

| Variable | Description | Default |
| --- | --- | --- |
| `SM_VLLM_MODEL` | Model ID or path (auto-detected from `/opt/ml/model` or `HF_MODEL_ID` if unset) | — |
| `SM_VLLM_TENSOR_PARALLEL_SIZE` | Number of GPUs | `1` |
| `SM_VLLM_MAX_MODEL_LEN` | Maximum sequence length | Model default |
| `SM_VLLM_GPU_MEMORY_UTILIZATION` | Fraction of GPU memory to use | `0.9` |
| `SM_VLLM_ENFORCE_EAGER` | Disable CUDA graphs | `false` |
| `SM_VLLM_TRUST_REMOTE_CODE` | Allow custom model code | `false` |
| `HF_MODEL_ID` | Hugging Face model ID (fallback when `SM_VLLM_MODEL` is unset and `/opt/ml/model` is empty) | — |
| `HF_TOKEN` | Hugging Face token for gated models | — |
| `VLLM_ENABLE_CUDA_COMPATIBILITY` | Enable CUDA 13 forward compatibility for hosts with older NVIDIA drivers | `0` |

## Known Limitations

- **CosyVoice3 requires `--trust-remote-code` and ~32 GB host RAM during model load.** Use `g6e.xlarge` or larger.
- **Stable-Audio-Open output is capped at ~47 seconds per request** by the model itself. For longer clips, run multiple requests and concatenate
  client-side.
- **First-request latency on SageMaker.** TTS, audio, and video models can exceed the 60s real-time invoke timeout due to `torch.compile` warmup. Use
  async inference or retry after warmup.

## Full Reference

- [vLLM-Omni Documentation](https://github.com/vllm-project/vllm-omni)
- [vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args.html)
