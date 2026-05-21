# EC2 Deployment

The container runs `vllm serve --omni` and exposes the OpenAI-compatible API on port 8080. Any `vllm serve` flag may be appended to `docker run`. See
[Configuration](../configuration.md) for the full list.

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

### Voice Cloning

Use [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) or
[CosyVoice3-0.5B](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) and pass a reference audio clip plus its exact transcript. CosyVoice3
additionally requires `--trust-remote-code` and a host with ≥ 32 GB RAM.

```bash
REF_AUDIO_B64=$(base64 -w 0 < reference_voice.wav)

curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": \"Hello, this is a cloned voice.\",
    \"ref_audio\": \"${REF_AUDIO_B64}\",
    \"ref_text\": \"The exact transcript of reference_voice.wav goes here.\",
    \"language\": \"English\"
  }" --output cloned.wav
```

`ref_text` MUST be the **exact** transcript of the reference clip. A mismatched transcript causes Code2Wav to emit malformed output (upstream
[vllm-omni#3124](https://github.com/vllm-project/vllm-omni/issues/3124)).

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

### Sync (returns MP4 directly)

```bash
curl http://localhost:8080/v1/videos/sync \
  -F "prompt=a dog running on a beach" \
  -F "num_frames=17" -F "num_inference_steps=4" \
  -F "size=480x320" -F "seed=42" \
  --output video.mp4
```

### Async (job-ID polling)

Use async when you want to fire-and-forget long generations or need to overlap multiple jobs.

```bash
# Submit
JOB_ID=$(curl -s -X POST http://localhost:8080/v1/videos \
  -F "prompt=a dog running on a beach" \
  -F "num_frames=17" -F "num_inference_steps=30" \
  -F "size=480x320" -F "seed=42" | jq -r '.id')

# Poll until done
while [ "$(curl -s http://localhost:8080/v1/videos/${JOB_ID} | jq -r '.status')" != "completed" ]; do
  sleep 5
done

# Download
curl -s http://localhost:8080/v1/videos/${JOB_ID}/content --output video.mp4
```

Both routes accept the same form fields. The sync route is required for SageMaker real-time endpoints — async writes only the job-ID JSON to S3, not
the MP4.

## Multimodal Chat

**Model:** [Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) — requires ≥ 4 GPUs (e.g., `g6.12xlarge`, `g6e.12xlarge`). The talker stage
fails to load on single-GPU hosts.

```bash
docker run --gpus all --ipc=host -p 8080:8080 \
  public.ecr.aws/deep-learning-containers/vllm:omni-cuda \
  --model Qwen/Qwen2.5-Omni-3B \
  --tensor-parallel-size 4 \
  --max-model-len 16384 --dtype bfloat16
```

### Speech Output Requirements

For audio responses, three fields MUST be set correctly — the server returns 200 OK but produces empty or noisy audio if any is wrong:

1. `"modalities": ["audio"]` — using `["text", "audio"]` returns empty audio
2. `"sampling_params_list"` — a 3-element list (thinker, talker, code2wav). Built-in defaults produce noise
3. The exact Qwen system prompt (verbatim from the model card)

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Omni-3B",
    "modalities": ["audio"],
    "sampling_params_list": [
      {"temperature":0.0,"top_p":1.0,"top_k":-1,"max_tokens":2048,"seed":42,"detokenize":true,"repetition_penalty":1.1},
      {"temperature":0.9,"top_p":0.8,"top_k":40,"max_tokens":2048,"seed":42,"detokenize":true,"repetition_penalty":1.05,"stop_token_ids":[8294]},
      {"temperature":0.0,"top_p":1.0,"top_k":-1,"max_tokens":2048,"seed":42,"detokenize":true,"repetition_penalty":1.1}
    ],
    "messages": [
      {"role":"system","content":[{"type":"text","text":"You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
      {"role":"user","content":[{"type":"text","text":"Tell me a short bedtime story."}]}
    ]
  }' | jq -r '.choices[0].message.audio.data' | base64 -d > story.wav
```

For text-only output, omit `modalities` and `sampling_params_list`:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-Omni-3B", "messages": [{"role": "user", "content": "Say hello in one sentence."}], "max_tokens": 64}'
```

## Model-Specific Tuning

For recommended serving flags and hardware configurations, see [recipes.vllm.ai](https://recipes.vllm.ai/).
