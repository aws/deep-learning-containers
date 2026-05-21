# Supported Models

All models listed below are regression-tested on every DLC vLLM-Omni release and work with the images listed on the [Overview](../index.md) page.

The **Coverage** column indicates test depth: *Smoke* runs on every PR; *Benchmark* runs throughput and latency tests with pass/fail thresholds before
release. A *Smoke + Benchmark* tag means both apply.

## Tested Models

| Modality | Model | Coverage |
| --- | --- | --- |
| **TTS (preset voice)** | [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | Smoke + Benchmark |
| **TTS (voice clone)** | [Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | Smoke + Benchmark |
|  | [FunAudioLLM/CosyVoice3-0.5B](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) | Smoke + Benchmark |
| **Image generation** | [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | Smoke + Benchmark |
|  | [baidu/ERNIE-Image-Turbo](https://huggingface.co/baidu/ERNIE-Image-Turbo) | Smoke + Benchmark |
| **Video — text-to-video** | [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) | Smoke + Benchmark |
| **Video — unified create/edit** | [Wan-AI/Wan2.1-VACE-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-Diffusers) | Smoke + Benchmark |
| **Audio generation** | [stabilityai/stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) | Smoke + Benchmark |
| **Omni chat** | [Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) | Benchmark |

The Wan2.1-VACE model accepts text plus optional video, mask, or reference image inputs for unified video creation and editing — distinct from the
text-only Wan2.1-T2V pipeline.

## Model Compatibility

Any model supported by upstream vLLM-Omni should work. Requirements:

- Models must have a standard HuggingFace `config.json` with a recognized `model_type`, or be diffusers pipeline models with `model_index.json`.
- Multi-stage omni models (thinker + talker + decoder) like Qwen2.5-Omni need significantly more VRAM than the model size suggests. Refer to
  individual model cards for minimum GPU requirements.
