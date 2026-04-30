# vLLM-Omni Benchmarks

Lightweight async benchmark clients for vLLM-Omni multimodal endpoints. Each
client talks directly to the OpenAI-compatible HTTP endpoint on port 8080
(not the SageMaker `/invocations` middleware) so measurements reflect raw
model performance.

Driven by:

- **Workflow**: `.github/workflows/dispatch-vllm-omni-benchmark.yml`
- **Config**: `.github/config/model-tests/vllm-omni-model-tests.yml` (`benchmark:` section)
- **Dispatcher**: `test/vllm-omni/scripts/vllm_omni_benchmark_test.sh`

## Clients

| Script                          | Endpoint                          | Example models                 | Metrics                                              |
| ------------------------------- | --------------------------------- | ------------------------------ | ---------------------------------------------------- |
| `tts_benchmark_client.py`       | `POST /v1/audio/speech`           | qwen3-tts (CustomVoice + Base) | TTFB, E2E, audio duration, RTF, req/s                |
| `image_benchmark_client.py`     | `POST /v1/images/generations`     | flux2-klein-4b                 | E2E, images/s                                        |
| `video_benchmark_client.py`     | `POST /v1/videos` + poll          | wan2.1-t2v-1.3b                | submit latency, server inference time, E2E, videos/s |
| `chat_omni_benchmark_client.py` | `POST /v1/chat/completions` (SSE) | qwen2.5-omni-3b                | **TTFT**, **TPOT**, ITL, E2E, req/s, output tokens/s |

### Why SSE for chat-omni?

The chat client uses **streaming** (`stream=true`) because the most important
user-perceived metric for chat is **TTFT** (Time To First Token) — how long
until the user sees *any* output. TTFT is only measurable when the server
streams tokens back incrementally via SSE `data: {...}\n\n` chunks.

The client records:

- `ttft_ms` — from request send to first `delta.content`
- `tpot_ms` — `(e2e - ttft) / (output_tokens - 1)` for subsequent tokens
- `itl_ms_mean` — per-pair inter-token latency
- `output_tokens_per_second` — aggregate throughput

Metrics match `vllm bench serve --omni --backend openai-chat-omni` so numbers
are directly comparable with upstream vllm-omni benchmarks.

### Why custom clients instead of `vllm bench serve`?

1. `vllm bench serve --omni` only targets `/v1/chat/completions`; it can't
   benchmark TTS speech, image gen, or video gen endpoints.
1. It requires `vllm-omni` installed on the invoking host, which complicates
   the CI runner environment.
1. Keeping all 4 modality clients in the same shape lets a single dispatcher
   - reporter handle every benchmark type.

## Running locally

```bash
pip install aiohttp

# Start the server (same pattern as EC2 smoke test)
docker run -d --gpus all --shm-size=10g \
  -v $(pwd)/models:/models -p 8080:8080 \
  <vllm-omni-image> \
  --model /models/<model-name> --port 8080 --stage-init-timeout 900

# Run a client
python3 tts_benchmark_client.py \
  --base-url http://localhost:8080 \
  --num-prompts 20 --concurrency 4 \
  --voice vivian --language English \
  --output-json /tmp/tts.json

# For voice-cloning (Base) models, provide ref audio + transcript
python3 tts_benchmark_client.py \
  --base-url http://localhost:8080 \
  --num-prompts 20 --concurrency 4 \
  --task-type Base --ref-audio ref.wav --ref-text "Hello, how are you?" \
  --output-json /tmp/tts_base.json

# Image
python3 image_benchmark_client.py \
  --base-url http://localhost:8080 \
  --num-prompts 8 --concurrency 1 --size 512x512 \
  --output-json /tmp/image.json

# Video (async submit + poll; skips content download by default)
python3 video_benchmark_client.py \
  --base-url http://localhost:8080 \
  --num-prompts 6 --concurrency 1 \
  --num-frames 17 --num-inference-steps 4 --size 480x320 \
  --output-json /tmp/video.json

# Chat (SSE streaming)
python3 chat_omni_benchmark_client.py \
  --base-url http://localhost:8080 \
  --num-prompts 16 --concurrency 2 \
  --max-tokens 128 --ignore-eos \
  --output-json /tmp/chat.json
```

## Reference audio fixture for Base (voice-cloning) TTS

The `tts-base` benchmark needs a reference WAV. Canonical copy in S3:

- `s3://dlc-cicd-models/test-fixtures/audio/tts_ref_vivian.wav` (256 KB, mono 24kHz)
- `s3://dlc-cicd-models/test-fixtures/audio/tts_ref_vivian.txt` (transcript)

The dispatcher (`vllm_omni_benchmark_test.sh`, `tts-base` case) downloads it
from S3 at run time — no local copy is needed.

## Output JSON shape

Each client writes `{"summary": {...}, "per_request": [...]}`. The `summary`
block is what `benchmark_report.py` aggregates into a markdown table for
`$GITHUB_STEP_SUMMARY`. Key fields by client:

- **TTS**: `requests_per_second`, `audio_throughput_s_per_s`, `ttfb_ms.{mean,median,p95,p99}`, `e2e_ms.*`, `rtf.*`
- **Image**: `requests_per_second`, `images_per_second`, `e2e_ms.*`
- **Video**: `requests_per_second`, `videos_per_second`, `submit_ms.*`, `server_inference_time_s.*`, `e2e_ms.*`
- **Chat**: `requests_per_second`, `output_tokens_per_second`, `ttft_ms.*`, `tpot_ms.*`, `itl_ms_mean.*`, `e2e_ms.*`

## Thresholds and CI

The `benchmark:` section of `vllm-omni-model-tests.yml` declares per-model
thresholds via `benchmark_config`. Supported threshold keys per type:

| Type              | Keys                                                                                |
| ----------------- | ----------------------------------------------------------------------------------- |
| `tts`, `tts-base` | `min_rps`, `min_audio_rtf_mult`, `max_p95_e2e_ms`                                   |
| `image`           | `min_images_per_s`, `max_p95_e2e_ms`                                                |
| `video`           | `min_videos_per_s`, `max_p95_e2e_ms`                                                |
| `chat`            | `min_rps`, `min_output_tps`, `max_p95_ttft_ms`, `max_p95_tpot_ms`, `max_p95_e2e_ms` |

Missing keys are skipped (no enforcement).

## Adding a new model

1. Add entry under `benchmark.codebuild-fleet` in `vllm-omni-model-tests.yml`
   with `name`, `s3_model`, `fleet`, `benchmark_type`, `benchmark_config`.
1. Ensure the model tarball is uploaded to `s3://dlc-cicd-models/omni-models/`.
1. Trigger `vLLM-Omni Benchmark Tests` workflow with the target image URI.
1. First run's output (in `$GITHUB_STEP_SUMMARY`) gives real numbers; tighten
   thresholds in the YAML if needed.
