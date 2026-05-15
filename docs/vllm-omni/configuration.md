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

For gated models, pass `-e HF_TOKEN=<your_hf_token>`.

## Amazon SageMaker AI (`omni-sagemaker-cuda`)

Set `SM_VLLM_*` environment variables on the container. Each is converted to the corresponding vLLM flag.

| Variable | Description | Default |
| --- | --- | --- |
| `SM_VLLM_MODEL` | Model ID or path (required) | — |
| `SM_VLLM_TENSOR_PARALLEL_SIZE` | Number of GPUs | `1` |
| `SM_VLLM_MAX_MODEL_LEN` | Maximum sequence length | Model default |
| `SM_VLLM_GPU_MEMORY_UTILIZATION` | Fraction of GPU memory to use | `0.9` |
| `SM_VLLM_ENFORCE_EAGER` | Disable CUDA graphs | `false` |
| `HF_TOKEN` | Hugging Face token for gated models | — |

## Known Limitations

- **Voice-clone TTS (Qwen3-TTS-Base) is slower in v1.1 than v1.0** due to an upstream Code2Wav decode-chunk un-batching regression. Preset-voice TTS is unaffected. Fix is merged upstream and will land in the next release.
- **CosyVoice3 requires `--trust-remote-code` and ~32 GB host RAM during model load.** Use `g6e.xlarge` or larger.
- **Stable-Audio-Open output is capped at ~47 seconds per request** by the model itself. For longer clips, run multiple requests and concatenate client-side.
- **First-request latency on SageMaker.** TTS, audio, and video models can exceed the 60s real-time invoke timeout due to `torch.compile` warmup. Use async inference or retry after warmup.

## Full Reference

- [vLLM-Omni Documentation](https://github.com/vllm-project/vllm-omni)
- [vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args.html)
