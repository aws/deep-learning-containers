# Qwen2.5-Omni-3B on EC2 GPU

Run [Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) (multimodal-in / text + speech-out) using the vLLM-Omni container — both as a
local (offline) server and as a remote (online) endpoint.

## Requirements

- **EC2 GPU instance with ≥ 4 GPUs**:
  - `g5.12xlarge` / `g6.12xlarge` (4× A10G, 24 GB each) — tested
  - `g6e.12xlarge` (4× L40S, 48 GB each) — preferred when available
- Amazon Linux 2023 with NVIDIA driver, Docker, and `nvidia-container-toolkit` (AWS Deep Learning AMIs include these)
- AWS credentials with ECR pull permission for `763104351884`
- Outbound internet to HuggingFace (first run downloads ~6 GB)

!!! note "Single-GPU note" Qwen2.5-Omni-3B's default stage layout puts the talker on GPU 1. On a single-GPU instance it fails or produces distorted
audio. Use a 4-GPU instance.

## One-time setup

```bash
# ECR login
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com

docker pull {{ images.latest_vllm_omni_ec2 }}

mkdir -p ~/hf-cache
```

## Start the server

```bash
docker run -d --name omni3b \
  --gpus all --shm-size=16g -p 8080:8080 \
  -v ~/hf-cache:/root/.cache/huggingface \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  {{ images.latest_vllm_omni_ec2 }} \
    Qwen/Qwen2.5-Omni-3B \
    --host 0.0.0.0 --port 8080 \
    --max-model-len 16384 --dtype bfloat16
```

First start takes ~8 minutes (weight download + 3-stage model load). Wait for ready:

```bash
until curl -sf http://localhost:8080/health >/dev/null; do sleep 10; done
echo ready
```

Stop and remove:

```bash
docker stop omni3b && docker rm omni3b
```

## Getting clean audio out

Three things are **required** on `/v1/chat/completions` to produce usable speech from Qwen2.5-Omni-3B:

1. `"modalities": ["audio"]`
2. `"sampling_params_list"` — a 3-element list (thinker, talker, code2wav). The image's built-in per-stage defaults are wrong and produce noise. Use
   the values shown below (from the official Qwen docs).
3. The exact Qwen system prompt.

!!! warning "Omitting `sampling_params_list` produces noise even though HTTP returns 200 with valid WAV bytes."

### Working curl

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
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
      {"role":"user","content":[{"type":"text","text":"Tell me a short, calming bedtime lullaby story for a 6-year-old girl."}]}
    ]
  }' | jq -r '.choices[0].message.audio.data' | base64 -d > out.wav
```

## Offline inference (on the GPU instance)

```python
--8<-- "examples/vllm-omni/qwen2.5-omni/offline_inference.py"
```

Run it:

```bash
python3 offline_inference.py
aplay out/lullaby.wav   # afplay on macOS
```

## Online inference (from a remote client)

Open TCP 8080 in the EC2 security group to your client IP, then:

```bash
export OMNI_ENDPOINT=http://ec2-xx-xx-xx-xx.us-west-2.compute.amazonaws.com:8080
python3 online_inference.py
```

```python
--8<-- "examples/vllm-omni/qwen2.5-omni/online_inference.py"
```

## API overview

OpenAI-compatible endpoints exposed by the container:

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/chat/completions` | Text / multimodal in → text or audio out (see above for audio) |
| `POST /v1/audio/speech` | Direct text-to-speech shortcut (voices: `Chelsie`, `Ethan`). ⚠️ In v1.0.0 the shortcut bypasses the thinker and does not apply the correct sampling params, producing noisy output. Prefer the chat route. |
| `GET /v1/audio/voices` | List voices |
| `GET /v1/models` | Show served model id |
| `GET /health` | Liveness |

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `NVMLError_InvalidArgument` in stage 1 during startup | Single-GPU instance — use a 4-GPU instance. |
| Audio sounds like noise/gibberish | Missing `sampling_params_list` — add it per above. |
| `message.audio: {}` empty on chat completions | Using `"modalities": ["text","audio"]`. Use `["audio"]` only. |
| `Cannot perform interactive login from non-TTY device` | AWS creds expired. Refresh `~/.aws/credentials` and re-run ECR login. |
| Health never goes 200 | Inspect `docker logs omni3b`. Weight download or OOM — need ≥4 GPUs with ≥24 GB each. |

## Costs (us-west-2, on-demand, April 2026)

- `g5.12xlarge` ≈ $5.67 / hour
- `g6e.12xlarge` ≈ $10.49 / hour

Stop the instance when idle; terminate to free EBS.
