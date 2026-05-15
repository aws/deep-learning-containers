# EC2 Deployment

The container runs `vllm serve --omni` and exposes the OpenAI-compatible API on port 8080. Any `vllm serve` flag may be appended to `docker run`. See [Configuration](../configuration.md) for the full list.

## Text-to-Speech

**Model:** [Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) — runs on a single 24 GB GPU.

```bash
docker run --gpus all -p 8080:8080 \
  public.ecr.aws/deep-learning-containers/vllm:omni-cuda \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

```bash
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, how are you?", "voice": "vivian", "language": "English"}' \
  --output speech.wav
```

For voice cloning, use [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) or [CosyVoice3-0.5B](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) with a reference audio clip.

## Audio Generation

**Model:** [Stable-Audio-Open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) — text-to-audio diffusion, runs on a single 24 GB GPU.

```bash
docker run --gpus all -p 8080:8080 \
  public.ecr.aws/deep-learning-containers/vllm:omni-cuda \
  --model stabilityai/stable-audio-open-1.0 \
  --gpu-memory-utilization 0.9 --trust-remote-code --enforce-eager
```

```bash
curl http://localhost:8080/v1/audio/generate \
  -H "Content-Type: application/json" \
  -d '{"input": "The sound of a dog barking", "audio_length": 5.0, "guidance_scale": 7.0, "num_inference_steps": 50}' \
  --output audio.wav
```

## Image Generation

**Model:** [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) — runs on a single 24 GB GPU.

```bash
docker run --gpus all -p 8080:8080 \
  public.ecr.aws/deep-learning-containers/vllm:omni-cuda \
  --model black-forest-labs/FLUX.2-klein-4B
```

```bash
curl http://localhost:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a red apple on a white table", "size": "512x512", "n": 1}'
```

## Video Generation

**Model:** [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) — needs a 48 GB GPU (L40S).

Two route options:

- **Sync** (`/v1/videos/sync`) — blocks until complete, returns raw MP4. Simpler and SageMaker-compatible.
- **Async** (`/v1/videos`) — returns a job ID; poll until complete, then download MP4.

```bash
docker run --gpus all -p 8080:8080 \
  public.ecr.aws/deep-learning-containers/vllm:omni-cuda \
  --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers
```

```bash
curl http://localhost:8080/v1/videos/sync \
  -H "Content-Type: multipart/form-data" \
  -F "prompt=a dog running on a beach" \
  -F "num_frames=17" -F "num_inference_steps=4" \
  -F "size=480x320" -F "seed=42" \
  --output video.mp4
```

## Multimodal Chat

**Model:** [Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) — requires ≥ 4 GPUs.

```bash
docker run --gpus all --ipc=host -p 8080:8080 \
  public.ecr.aws/deep-learning-containers/vllm:omni-cuda \
  --model Qwen/Qwen2.5-Omni-3B \
  --tensor-parallel-size 4
```

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-Omni-3B", "messages": [{"role": "user", "content": "Say hello in one sentence."}], "max_tokens": 64}'
```

## Model-Specific Tuning

For recommended serving flags and hardware configurations, see [recipes.vllm.ai](https://recipes.vllm.ai/).
